"""
Train 3 specialized LightGBM classifiers (PURE SKILL VERSION), split by surface & distance:
  - turf_short  : 芝 <= 1600m  → 上がり偏差値重視
  - turf_long   : 芝 >= 1800m  → 血統×馬場・斤量比率重視
  - dirt        : ダート (all) → 馬体重・斤量重視

【重要】 'odds' / 'popularity' を**学習から完全除外**した純粋実力モデル。
EV = AI確率 × 実際のオッズ で期待値を算出し、買い判断はシミュレータが行う。

Each model uses Walk-forward validation (train<=2024, test>=2025).
"""
import pandas as pd
import numpy as np
import os
import sys
import re
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─── JRA/地方 会場コード → (surface, typical_distance) マッピング ─────
# 実データから確認した venue_code を使い、芝/ダートと代表距離で分類する
# JRA競馬場コード: 01=札幌 02=函館 03=福島 04=新潟 05=東京 06=中山
#                  07=中京 08=京都 09=阪神 10=小倉
# 地方: 30=門別 35=盛岡 36=水沢 43=名古屋 44=大井 45=川崎 46=浦和 47=船橋 48=金沢 55=佐賀 etc.

# JRAは主に芝・ダート両方あるが、重賞の距離は概ね:
#   - 中央重賞: マイル前後(1400-1800)と長距離(2000+)が混在
# 今回はレース番号(最終R付近=重賞)と場コードで大まかに分類する

DIRT_VENUES = {str(i) for i in range(30, 99)}   # 地方はすべてダート

def parse_race_type(race_id: str, race_info: str):
    """race_id の venue_code からsurface/distanceを推定"""
    race_id = str(race_id)
    if len(race_id) < 12:
        return 'turf', 2000

    venue_code = race_id[4:6]
    race_round = race_id[6:8]   # 回数（1〜5程度）
    race_day   = race_id[8:10]  # 開催日数
    race_no    = race_id[10:12] # レース番号（11=重賞が多い）

    # 地方競馬 → ダート
    if venue_code in DIRT_VENUES:
        return 'dirt', 1600

    # JRA: race_info に距離ヒントがある場合
    info = str(race_info) if pd.notna(race_info) else ''
    import re as _re
    m = _re.search(r'([芝ダ])(\d{3,4})m', info)
    if m:
        surface = 'turf' if m.group(1) == '芝' else 'dirt'
        return surface, int(m.group(2))

    # JRA: G1重賞の代表的な距離マッピング（venue + 主要G1は長距離傾向）
    # 東京(05)・中山(06)・阪神(09)・京都(08)で多くの重賞が行われる
    # 重賞コード11のレース → 芝・2000m超が多い（天皇賞・ジャパンカップなど）
    long_dist_venues = {'05', '06', '08', '09'}  # 東京・中山・京都・阪神
    short_dist_venues = {'07', '04', '10'}        # 中京・新潟・小倉（スプリント線）
    if venue_code in short_dist_venues:
        return 'turf', 1400
    return 'turf', 2000  # デフォルト: 芝長距離

def assign_model_type(surface: str, dist: int) -> str:
    if surface == 'dirt':
        return 'dirt'
    if dist <= 1600:
        return 'turf_short'
    return 'turf_long'

# ─── ハイパーパラメータ（モデル別最適化） ──────────────────────
PARAMS = {
    'turf_short': dict(
        n_estimators=200, learning_rate=0.03, num_leaves=63,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, class_weight='balanced',
        random_state=42
    ),
    'turf_long': dict(
        n_estimators=150, learning_rate=0.04, num_leaves=31,
        max_depth=5, subsample=0.9, colsample_bytree=0.7,
        min_child_samples=30, class_weight='balanced',
        random_state=42
    ),
    'dirt': dict(
        n_estimators=200, learning_rate=0.03, num_leaves=63,
        max_depth=6, subsample=0.85, colsample_bytree=0.75,
        min_child_samples=20, class_weight='balanced',
        random_state=42
    ),
}

# ─── 特徴量重みヒント（important_features を先頭に並べる） ─────
# LightGBM は特徴量の順序に依存しないが、明示的なログ用として記録
PRIORITY_FEATURES = {
    'turf_short': ['prev1_relative_last3f', 'prev2_relative_last3f', 'prev3_relative_last3f'],
    'turf_long':  ['sire_track_interaction', 'weight_ratio', 'bms'],
    'dirt':       ['horse_weight', 'weight_ratio', 'weight_delta'],
}

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    processed_parquet = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')
    raw_parquet       = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    model_dir         = os.path.join(base_dir, 'src', 'models')

    # ─── データ読み込み ──────────────────────────────────────────
    print("Loading processed data from Parquet cache ...")
    df = pd.read_parquet(processed_parquet)

    # race_info を raw から join して距離・馬場を取得
    raw = pd.read_parquet(raw_parquet)[['race_id', 'race_info']].drop_duplicates('race_id')
    raw['race_id'] = raw['race_id'].astype(str)
    df['race_id'] = df['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    # surface / distance / model_type を付与
    parsed = df.apply(
        lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1
    )
    df['surface'] = [p[0] for p in parsed]
    df['distance'] = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)

    print(f"\nRace type distribution:")
    print(df.drop_duplicates('race_id')['model_type'].value_counts())

    # ─── 目的変数と年列 ─────────────────────────────────────────
    df['is_top3'] = (df['rank'] <= 3).astype(int)
    df['year']    = df['race_id'].str[:4].astype(int)

    # オッズ・人気を学習から完全除外（純粋実力モデル）
    ODDS_COLS    = ['odds', 'popularity']
    ignore_cols  = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance', 'model_type', 'race_info'] + ODDS_COLS
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    print(f"\n[PURE MODEL] Feature count: {len(feature_cols)} (odds/popularity excluded)")
    print(f"  Features: {feature_cols}")

    results = {}

    for mtype in ['turf_short', 'turf_long', 'dirt']:
        sub = df[df['model_type'] == mtype].copy()
        train = sub[sub['year'] <= 2024]
        test  = sub[sub['year'] >= 2025]

        if len(train) < 100 or len(test) < 30:
            print(f"\n[SKIP] {mtype}: not enough data (train={len(train)}, test={len(test)})")
            continue

        print(f"\n{'='*55}")
        print(f"  Model: {mtype.upper()}")
        print(f"  Train (<=2024): {len(train)} rows | Test (>=2025): {len(test)} rows")
        pf = PRIORITY_FEATURES[mtype]
        print(f"  Priority features: {pf}")
        print(f"{'='*55}")

        X_train = train[feature_cols]
        y_train = train['is_top3']
        X_test  = test[feature_cols]
        y_test  = test['is_top3']

        model = lgb.LGBMClassifier(**PARAMS[mtype])
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(0)])

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        ll  = log_loss(y_test, y_proba)

        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Test ROC AUC  : {auc:.4f}")
        print(f"  Test Log Loss : {ll:.4f}")

        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print(f"  Top-10 features:")
        for feat, val in imp.head(10).items():
            marker = " ★" if feat in pf else ""
            print(f"    {feat:<35} {val:>6}{marker}")

        # モデル保存（pure版は専用ファイル名）
        save_path = os.path.join(model_dir, f'lgbm_{mtype}_pure.pkl')
        joblib.dump(model, save_path)
        print(f"  Saved => {save_path}")

        results[mtype] = {
            'model': model,
            'feature_cols': feature_cols,
            'test_df': test.copy(),
            'auc': auc
        }

    return results

if __name__ == "__main__":
    main()
