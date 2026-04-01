"""
ダート専用 最終決戦モデル（surface == 'dirt'）
プランJ（実力純粋モデル）構成
1. ダートデータのみで専用学習（オッズ・人気除外）
2. 騎手×競馬場ターゲットエンコーディング（JV）
3. 馬体重・斤量・枠番の相関を重視するハイパーパラメータ設定
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


def add_dirt_specific_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    ダート専用の追加特徴量:
    - 枠番 × 砂被り指数: 内枠ほど砂を被るリスクあり(地方はその逆も)
    - 体重×斤量の複合スコア（重い馬は斤量負担が相対的に小さい）
    - venue別の枠番有利係数（地方と中央でラチの内外有利が異なる）
    """
    df = df.copy()

    # 枠番を数値化
    df['bracket_num'] = pd.to_numeric(df['bracket'], errors='coerce').fillna(4)

    # 内枠（1-2）と外枠フラグ（7-8相当）
    df['is_inner_bracket'] = (df['bracket_num'] <= 2).astype(int)
    df['is_outer_bracket'] = (df['bracket_num'] >= 7).astype(int)

    # 馬体重×斤量 複合スコア（重い馬ほど斤量負担が軽くなると考える）
    df['horse_weight_raw'] = pd.to_numeric(
        df['horse_weight'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    ).fillna(480)
    df['weight_burden_score'] = df['weight_constraint'] / (df['horse_weight_raw'] + 1e-5)

    # ダート地方/中央フラグ
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df['is_local_dirt'] = df['venue_code'].astype(int).apply(lambda v: 1 if v >= 30 else 0)

    # 地方ダートの場合: 内枠が有利（内ラチ沿いで砂被り少）
    # JRAダートの場合: 外枠気味が有利（砂を直接被らない）
    df['dirt_bracket_score'] = np.where(
        df['is_local_dirt'] == 1,
        (3 - df['bracket_num']) * 0.02,        # 地方: 内枠ほど+
        (df['bracket_num'] - 4.5) * 0.015      # JRA: 外枠ほど+
    )

    # venue × bracket の過去成績をターゲットエンコーディング
    train_df = df[train_mask].copy()
    train_df['is_win'] = (train_df['rank'] == 1.0).astype(float)

    bracket_venue_stats = train_df.groupby(['venue_code', 'bracket_num']).agg(
        bv_win_rate = ('is_win', 'mean'),
        bv_count    = ('is_win', 'count'),
    ).reset_index()

    global_win = train_df['is_win'].mean()
    bracket_venue_stats['bv_win_rate_adj'] = (
        bracket_venue_stats['bv_win_rate'] * bracket_venue_stats['bv_count'] + global_win * 5
    ) / (bracket_venue_stats['bv_count'] + 5)

    df = df.merge(bracket_venue_stats[['venue_code', 'bracket_num', 'bv_win_rate_adj']],
                  on=['venue_code', 'bracket_num'], how='left')
    df['bv_win_rate_adj'] = df['bv_win_rate_adj'].fillna(global_win)

    return df


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

    # ダートのみ抽出
    sub = df[df['model_type'] == 'dirt'].copy()
    sub['year']    = sub['race_id'].str[:4].astype(int)
    sub['is_top3'] = (sub['rank'] <= 3).astype(int)

    train_mask = sub['year'] <= 2024
    print(f"\n[DIRT SPECIALIST]")
    print(f"  Train (<=2024): {train_mask.sum()} rows")
    print(f"  Test  (>=2025): {(~train_mask).sum()} rows")

    # ダート専用特徴量追加
    print("  Adding Dirt-specific features (bracket bias, weight burden score) ...")
    sub = add_dirt_specific_features(sub, train_mask)
    sub = sub.reset_index(drop=True)
    train_mask = sub['year'] <= 2024   # インデックスリセット後に再生成

    # 騎手×競馬場ターゲットエンコーディング
    print("  Adding Jockey × Venue target encoding ...")
    sub = add_jockey_venue_encoding(sub, train_mask)
    sub = sub.reset_index(drop=True)

    # 特徴量定義（オッズ・人気除外）
    BASE_EXCLUDE = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance',
                    'model_type', 'race_info', 'odds', 'popularity', 'venue_code',
                    'jockey', 'bracket',       # 数値エンコード版で代替
                    'horse_weight']            # horse_weight_raw で代替
    feature_cols = [c for c in sub.columns if c not in BASE_EXCLUDE]

    TE_FEATS   = [f for f in feature_cols if 'jv_' in f or 'j_win' in f or 'j_top' in f or 'bv_' in f]
    DIRT_FEATS = [f for f in feature_cols if any(k in f for k in
                  ['weight_burden', 'is_inner', 'is_outer', 'dirt_bracket', 'weight_ratio', 'weight_delta'])]

    print(f"  Feature count: {len(feature_cols)}")
    print(f"  JV-TE features   : {TE_FEATS}")
    print(f"  Dirt-spec features: {DIRT_FEATS}")

    X_train = sub[sub['year'] <= 2024][feature_cols]
    y_train = sub[sub['year'] <= 2024]['is_top3']
    X_test  = sub[sub['year'] >= 2025][feature_cols]
    y_test  = sub[sub['year'] >= 2025]['is_top3']

    # ─── ハイパーパラメータ（ダート最適化版）─────────────────────
    # ダート: 馬体重と斤量の相関パターンが芝より線形的 → 木の深さより枚数重視
    BEST_PARAMS = dict(
        n_estimators     = 600,    # 多めにしてアンサンブル安定化
        learning_rate    = 0.015,  # 低めLR で精緻な学習
        num_leaves       = 63,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.7,
        min_child_samples= 20,
        reg_alpha        = 0.15,
        reg_lambda       = 0.8,
        class_weight     = 'balanced',
        random_state     = 42,
    )

    print(f"\n  Training LightGBM DIRT ...")
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
    print(f"  Test ROC AUC  : {auc:.4f}  (前回純粋モデル: 0.6640 → 改善幅: {auc-0.6640:+.4f})")
    print(f"  Test Log Loss : {ll:.4f}")

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top-15 features:")
    for feat, val in imp.head(15).items():
        tag = "🆕" if feat in TE_FEATS else ("⛏️" if feat in DIRT_FEATS else "")
        print(f"    {feat:<40} {val:>6}  {tag}")

    # 保存
    save_path = os.path.join(model_dir, 'lgbm_dirt_final.pkl')
    meta_path = os.path.join(model_dir, 'lgbm_dirt_final_meta.pkl')
    joblib.dump(model, save_path)
    joblib.dump({'feature_cols': feature_cols, 'auc': auc}, meta_path)
    print(f"\n  Saved => {save_path}")
    print(f"  Meta  => {meta_path}")


if __name__ == "__main__":
    main()
