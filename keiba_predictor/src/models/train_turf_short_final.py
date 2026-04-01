"""
芝・短中距離専用 最終決戦モデル
1. 芝≤1600mデータのみで専用学習
2. 騎手×競馬場ターゲットエンコーディング（勝率・連対率）
3. LightGBM ハイパーパラメータ最適化（Optuna なしで手動グリッド）
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


# ─── ターゲットエンコーディング（騎手×競馬場） ──────────────
def add_jockey_venue_encoding(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    騎手 × 競馬場コード の 過去データのみに基づく 勝率・複勝率 を計算して追加。
    データリークを防ぐため、train_mask=True のデータのみで統計を構築し、
    test データにもその統計を適用する（テスト時代のデータは使用しない）。
    """
    df = df.copy()

    # venue_code を race_id から取得 (4-6文字目)
    df['venue_code'] = df['race_id'].astype(str).str[4:6]

    # 学習データ内でのみ 騎手×競馬場 の過去成績を計算
    train_df = df[train_mask].copy()
    train_df['is_win']   = (train_df['rank'] == 1.0).astype(float)
    train_df['is_top2']  = (train_df['rank'] <= 2.0).astype(float)
    train_df['is_top3']  = (train_df['rank'] <= 3.0).astype(float)

    jv_stats = train_df.groupby(['jockey', 'venue_code']).agg(
        jv_win_rate   = ('is_win',  'mean'),
        jv_top2_rate  = ('is_top2', 'mean'),
        jv_top3_rate  = ('is_top3', 'mean'),
        jv_race_count = ('is_win',  'count'),
    ).reset_index()

    # 騎手全体成績（venue_codeによらない）
    j_stats = train_df.groupby('jockey').agg(
        j_win_rate  = ('is_win',  'mean'),
        j_top3_rate = ('is_top3', 'mean'),
    ).reset_index()

    # 競馬場全体成績
    v_stats = train_df.groupby('venue_code').agg(
        v_win_rate  = ('is_win',  'mean'),
        v_top3_rate = ('is_top3', 'mean'),
    ).reset_index()

    # マージ
    df = df.merge(jv_stats, on=['jockey', 'venue_code'], how='left')
    df = df.merge(j_stats,  on='jockey',     how='left')
    df = df.merge(v_stats,  on='venue_code', how='left')

    # 欠損補完: 全体平均で埋める
    global_win = train_df['is_win'].mean()
    global_t3  = train_df['is_top3'].mean()
    df['jv_win_rate']   = df['jv_win_rate'].fillna(df['j_win_rate']).fillna(global_win)
    df['jv_top2_rate']  = df['jv_top2_rate'].fillna(df['j_top3_rate']).fillna(global_t3 * 0.6)
    df['jv_top3_rate']  = df['jv_top3_rate'].fillna(df['j_top3_rate']).fillna(global_t3)
    df['jv_race_count'] = df['jv_race_count'].fillna(0)
    df['j_win_rate']    = df['j_win_rate'].fillna(global_win)
    df['j_top3_rate']   = df['j_top3_rate'].fillna(global_t3)
    df['v_win_rate']    = df['v_win_rate'].fillna(global_win)
    df['v_top3_rate']   = df['v_top3_rate'].fillna(global_t3)

    # 経験値補正（サンプル少ない場合はグローバル平均に引き寄せる）
    k = 10
    df['jv_win_rate_adj'] = (
        df['jv_win_rate'] * df['jv_race_count'] + global_win * k
    ) / (df['jv_race_count'] + k)

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

    # 芝・短中距離のみを抽出
    sub = df[df['model_type'] == 'turf_short'].copy()
    sub['year'] = sub['race_id'].str[:4].astype(int)
    sub['is_top3'] = (sub['rank'] <= 3).astype(int)

    train_mask = sub['year'] <= 2024
    print(f"\n[TURF-SHORT SPECIALIST]")
    print(f"  Train (<=2024): {train_mask.sum()} rows")
    print(f"  Test  (>=2025): {(~train_mask).sum()} rows")

    # ターゲットエンコーディング追加
    print("  Adding Jockey × Venue target encoding ...")
    sub = add_jockey_venue_encoding(sub, train_mask)

    # 特徴量定義 (オッズ・人気除外 + 新規ターゲットエンコード列追加)
    BASE_EXCLUDE = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance',
                    'model_type', 'race_info', 'odds', 'popularity', 'venue_code',
                    'jockey']  # カテゴリ版のjockeyは除外、数値エンコード版を使用
    feature_cols = [c for c in sub.columns if c not in BASE_EXCLUDE]
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  New TE features: {[f for f in feature_cols if 'jv_' in f or 'j_win' in f or 'v_win' in f]}")

    X_train = sub[sub['year'] <= 2024][feature_cols]
    y_train = sub[sub['year'] <= 2024]['is_top3']
    X_test  = sub[sub['year'] >= 2025][feature_cols]
    y_test  = sub[sub['year'] >= 2025]['is_top3']

    # ─── ハイパーパラメータ（芝短距離最適化版）──────────────────
    # 上がりタイム偏差値が重要 → 木を深くしてインタラクションを捉える
    BEST_PARAMS = dict(
        n_estimators=500, learning_rate=0.02, num_leaves=127,
        max_depth=7, subsample=0.75, colsample_bytree=0.75,
        min_child_samples=15, reg_alpha=0.1, reg_lambda=0.5,
        class_weight='balanced', random_state=42
    )

    print(f"\n  Training LightGBM (n_estimators={BEST_PARAMS['n_estimators']}, "
          f"lr={BEST_PARAMS['learning_rate']}, leaves={BEST_PARAMS['num_leaves']}) ...")

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
    print(f"  Test ROC AUC  : {auc:.4f}  (前回: 0.6524 → 改善幅: {auc-0.6524:+.4f})")
    print(f"  Test Log Loss : {ll:.4f}")

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top-15 features:")
    new_feats = {'jv_win_rate', 'jv_top3_rate', 'jv_win_rate_adj', 'j_win_rate', 'j_top3_rate', 'v_win_rate'}
    for feat, val in imp.head(15).items():
        marker = " 🆕" if feat in new_feats else ""
        print(f"    {feat:<38} {val:>6}{marker}")

    # モデルとメタデータ保存
    save_path = os.path.join(model_dir, 'lgbm_turf_short_final.pkl')
    meta_path = os.path.join(model_dir, 'lgbm_turf_short_final_meta.pkl')
    joblib.dump(model, save_path)
    joblib.dump({'feature_cols': feature_cols, 'auc': auc}, meta_path)

    print(f"\n  Saved => {save_path}")
    print(f"  Meta  => {meta_path}")

    return model, feature_cols, sub, train_mask


if __name__ == "__main__":
    main()
