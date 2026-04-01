"""
芝・中長距離専用 最終決戦モデル（distance >= 1800m）
プランJ（実力純粋モデル）構成
1. 芝≥1800mデータのみで専用学習（オッズ・人気除外）
2. 騎手×競馬場ターゲットエンコーディング（JV）
3. 血統×馬場条件の特徴量重視（sire_track_interaction, bms）
4. Walk-forward: train<=2024 / test>=2025
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding


def main():
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    proc_pq   = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')
    raw_pq    = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    model_dir = os.path.join(base_dir, 'src', 'models')

    print("Loading data from Parquet cache ...")
    df  = pd.read_parquet(proc_pq)
    raw = pd.read_parquet(raw_pq)[['race_id', 'race_info']].drop_duplicates('race_id')
    df['race_id']  = df['race_id'].astype(str)
    raw['race_id'] = raw['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    # race type
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)

    # 芝・中長距離のみを抽出
    sub = df[df['model_type'] == 'turf_long'].copy()
    sub['year']    = sub['race_id'].str[:4].astype(int)
    sub['is_top3'] = (sub['rank'] <= 3).astype(int)

    train_mask = sub['year'] <= 2024
    print(f"\n[TURF-LONG SPECIALIST]")
    print(f"  Train (<=2024): {train_mask.sum()} rows")
    print(f"  Test  (>=2025): {(~train_mask).sum()} rows")

    # ターゲットエンコーディング追加（train側のデータのみで構築）
    print("  Adding Jockey × Venue target encoding ...")
    sub = add_jockey_venue_encoding(sub, train_mask)

    # 特徴量定義（オッズ・人気除外）
    BASE_EXCLUDE = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance',
                    'model_type', 'race_info', 'odds', 'popularity', 'venue_code',
                    'jockey']  # 数値エンコード版を使うのでカテゴリjockeyは除外
    feature_cols = [c for c in sub.columns if c not in BASE_EXCLUDE]

    # 血統×馬場と騎手×競馬場TEを明示
    TE_FEATS  = [f for f in feature_cols if 'jv_' in f or 'j_win' in f or 'v_win' in f or 'j_top' in f]
    BLD_FEATS = [f for f in feature_cols if 'sire' in f or 'bms' in f or 'lineage' in f]
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  JV-TE features  : {TE_FEATS}")
    print(f"  Bloodline feats : {BLD_FEATS}")

    X_train = sub[sub['year'] <= 2024][feature_cols]
    y_train = sub[sub['year'] <= 2024]['is_top3']
    X_test  = sub[sub['year'] >= 2025][feature_cols]
    y_test  = sub[sub['year'] >= 2025]['is_top3']

    # ─── ハイパーパラメータ（中長距離最適化版）──────────────────
    # 中長距離: 血統・騎手・馬場条件などの組み合わせが重要
    # → 木をやや広く(leaves多め)、過学習防ぐためL1/L2正則化を効かせる
    BEST_PARAMS = dict(
        n_estimators     = 500,
        learning_rate    = 0.02,
        num_leaves       = 63,     # 短距離より少なくしてシンプルに
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        reg_alpha        = 0.2,    # L1 正則化強め（血統カテゴリの過学習防止）
        reg_lambda       = 1.0,    # L2 正則化
        class_weight     = 'balanced',
        random_state     = 42,
    )

    print(f"\n  Training LightGBM TURF-LONG ...")
    model = lgb.LGBMClassifier(**BEST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ll  = log_loss(y_test, y_proba)

    print(f"\n  ─── 専用モデル 評価結果 ───")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test ROC AUC  : {auc:.4f}  (前回純粋モデル: 0.6762 → 改善幅: {auc-0.6762:+.4f})")
    print(f"  Test Log Loss : {ll:.4f}")

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top-15 features:")
    new_feats = set(TE_FEATS)
    for feat, val in imp.head(15).items():
        tag = "🆕" if feat in new_feats else ("🩸" if feat in BLD_FEATS else "")
        print(f"    {feat:<38} {val:>6}  {tag}")

    # モデルとメタデータ保存
    save_path = os.path.join(model_dir, 'lgbm_turf_long_final.pkl')
    meta_path = os.path.join(model_dir, 'lgbm_turf_long_final_meta.pkl')
    joblib.dump(model, save_path)
    joblib.dump({'feature_cols': feature_cols, 'auc': auc}, meta_path)
    print(f"\n  Saved => {save_path}")
    print(f"  Meta  => {meta_path}")

    return model, feature_cols, sub


if __name__ == "__main__":
    main()
