"""
三冠馬 完全再構築スクリプト
- RAW parquetの'着 順'列（正確な着順1-15）を優先使用
- 全2941行（2023-2026/Q1）から正しくpreprocess
- AUC=1.0問題を修正（is_top3のラベルバランスを修正）
"""
import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')

PARAMS = {
    'turf_short': dict(n_estimators=300, learning_rate=0.03, num_leaves=31,
                       max_depth=5, min_child_samples=5, class_weight='balanced', random_state=42),
    'turf_long':  dict(n_estimators=400, learning_rate=0.025, num_leaves=31,
                       max_depth=5, min_child_samples=8, class_weight='balanced', random_state=42),
    'dirt':       dict(n_estimators=400, learning_rate=0.025, num_leaves=31,
                       max_depth=5, min_child_samples=8, class_weight='balanced', random_state=42),
}


def load_raw_data_correctly() -> pd.DataFrame:
    """
    RAW parquetを正確に読み込む。
    parquetには英語列(rank=全部1.0)と日本語列(着 順=1-15)の両方がある。
    必ず '着 順' を使用すること。
    """
    raw_pq = os.path.join(BASE_DIR, 'data', 'raw', 'scraped_results.parquet')
    raw = pd.read_parquet(raw_pq)

    print(f"RAW columns: {raw.columns.tolist()}")
    print(f"RAW shape: {raw.shape}")

    # '着 順'を最優先でrank確定（英語版rankは全て1.0で使えない）
    if '着 順' in raw.columns:
        rank_series = raw['着 順']
        # DataFrameになっていた場合は最初の列を使う
        if isinstance(rank_series, pd.DataFrame):
            rank_series = rank_series.iloc[:, 0]
        raw['rank'] = pd.to_numeric(rank_series.astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        print(f"着 順 → rank: {raw['rank'].value_counts().sort_index().to_dict()}")
    else:
        print("WARNING: '着 順'が見つかりません！ 'rank'列を使用します")
        raw['rank'] = pd.to_numeric(raw.get('rank', pd.Series(dtype=float)), errors='coerce')

    # 各列を確実に1列として取得する関数
    def get_col(df, name, default=None):
        if name not in df.columns:
            return pd.Series(default, index=df.index)
        col = df[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return col

    # 必要列を安全に取得（JP優先、なければ英語）
    out = pd.DataFrame()
    out['race_id']    = get_col(raw, 'race_id').astype(str)
    out['horse_id']   = get_col(raw, 'horse_id', '').astype(str)
    out['race_date']  = get_col(raw, 'race_date')
    out['race_info']  = get_col(raw, 'race_info', '')
    out['rank']       = raw['rank']
    out['weather']    = get_col(raw, 'weather', '晴').fillna('晴')
    out['track']      = get_col(raw, 'track', '良').fillna('良')
    out['passage_rank'] = get_col(raw, 'passage_rank')
    out['last3f_time']  = get_col(raw, 'last3f_time')
    out['odds']       = pd.to_numeric(get_col(raw, '単勝',  get_col(raw, 'odds')),  errors='coerce')
    out['popularity'] = pd.to_numeric(get_col(raw, '人 気', get_col(raw, 'popularity')), errors='coerce')
    out['horse_num']  = pd.to_numeric(get_col(raw, '馬 番', get_col(raw, 'horse_num')),  errors='coerce')
    out['weight_constraint'] = pd.to_numeric(get_col(raw, '斤量', get_col(raw, 'weight_constraint')), errors='coerce')
    out['horse_weight'] = get_col(raw, '馬体重', get_col(raw, 'horse_weight'))
    out['jockey']     = get_col(raw, '騎手', get_col(raw, 'jockey', '')).astype(str)
    out['trainer']    = get_col(raw, '調教師', get_col(raw, 'trainer', '')).astype(str)
    out['sex_age']    = get_col(raw, '性齢', get_col(raw, 'sex_age', '')).astype(str)
    out['horse_name'] = get_col(raw, '馬名', get_col(raw, 'horse_name', '')).astype(str)
    out['bracket']    = pd.to_numeric(get_col(raw, '枠 番', get_col(raw, 'bracket')), errors='coerce')

    return out


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を生成"""
    df['race_date_dt'] = pd.to_datetime(df['race_date'], errors='coerce')
    df['horse_weight_raw'] = df['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)

    # 時系列ソート → prev特徴量
    df = df.sort_values(['horse_id', 'race_date_dt']).reset_index(drop=True)
    df['prev_rank']  = df.groupby('horse_id')['rank'].shift(1)
    df['rest_days']  = (df['race_date_dt'] - df.groupby('horse_id')['race_date_dt'].shift(1)).dt.days
    df['weight_delta'] = df['weight_constraint'] - df.groupby('horse_id')['weight_constraint'].shift(1)
    df['weight_delta']  = df['weight_delta'].fillna(0)
    df['weight_ratio']  = df['weight_constraint'] / (df['horse_weight_raw'] + 1e-5)

    # 上がり偏差値
    df['last3f_time'] = pd.to_numeric(df['last3f_time'], errors='coerce')
    rm  = df.groupby('race_id')['last3f_time'].transform('mean')
    rs  = df.groupby('race_id')['last3f_time'].transform('std')
    df['relative_last3f'] = ((df['last3f_time'] - rm) / (rs + 1e-5) * 10 + 50).fillna(50)
    df['prev1_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(1).fillna(50)
    df['prev2_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(2).fillna(50)
    df['prev3_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(3).fillna(50)

    # 通過順位
    def parse_passage(p):
        if pd.isna(p) or not isinstance(p, str): return 7
        parts = [int(v) for v in str(p).split('-') if v.isdigit()]
        return parts[-1] if parts else 7
    df['passage_num'] = df['passage_rank'].apply(parse_passage) if 'passage_rank' in df.columns else 7
    df['prev1_passage_num'] = df.groupby('horse_id')['passage_num'].shift(1).fillna(7)
    df['prev2_passage_num'] = df.groupby('horse_id')['passage_num'].shift(2).fillna(7)
    df['prev3_passage_num'] = df.groupby('horse_id')['passage_num'].shift(3).fillna(7)

    # 血統情報
    ped_path = os.path.join(BASE_DIR, 'data', 'raw', 'horse_pedigree.csv')
    if os.path.exists(ped_path):
        ped = pd.read_csv(ped_path, dtype=str)
        ped['horse_id'] = ped['horse_id'].astype(str)
        df = df.merge(ped[['horse_id','sire','bms','lineage']], on='horse_id', how='left')
    else:
        df['sire'] = df['trainer']
        df['bms']  = df['jockey']
        df['lineage'] = 'unknown'
    df['sire']    = df['sire'].fillna('unknown').astype(str)
    df['bms']     = df['bms'].fillna('unknown').astype(str)
    df['lineage'] = df['lineage'].fillna('unknown').astype(str)
    df['sire_track_interaction'] = df['sire'] + '_' + df['track'].astype(str)

    df['is_long_trip'] = 0
    df['is_stay']      = 0

    # カテゴリエンコード（LabelEncoder）
    CAT_COLS = ['sex_age', 'jockey', 'trainer', 'weather', 'track',
                'sire', 'bms', 'lineage', 'sire_track_interaction']
    les = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            les[col] = le
    joblib.dump(les, os.path.join(BASE_DIR, 'src', 'features', 'encoders_v3.pkl'))

    # race type付与
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['venue_code'] = df['race_id'].str[4:6]
    df['year']       = df['race_id'].str[:4].astype(int)
    df['is_top3']    = (df['rank'] <= 3).astype(int)

    print(f"\nPreprocessed shape: {df.shape}")
    print(f"is_top3 分布: {df['is_top3'].value_counts().to_dict()}")
    print(f"model_type 分布:\n{df['model_type'].value_counts()}")
    return df


def train_model(mtype: str, df: pd.DataFrame):
    sub = df[df['model_type'] == mtype].copy()
    top3_dist = sub['is_top3'].value_counts().to_dict()
    print(f"\n{'='*55}")
    print(f"  {mtype}: {len(sub)} rows | is_top3: {top3_dist}")

    if sub['is_top3'].nunique() < 2:
        print(f"  [SKIP] ラベルが1種類のみ ({top3_dist}) — データ不足")
        return False

    if mtype == 'dirt':
        tmask = pd.Series([True]*len(sub), index=sub.index)
        sub = add_dirt_specific_features(sub, tmask).reset_index(drop=True)

    tmask = pd.Series([True]*len(sub), index=sub.index)
    sub   = add_jockey_venue_encoding(sub, tmask).reset_index(drop=True)

    EXCLUDE = {'rank','is_top3','race_id','year','surface','distance','model_type',
               'race_info','race_date','race_date_dt','odds','popularity',
               'venue_code','horse_id','horse_name','time','margin','passage_rank',
               'last3f_time','relative_last3f','passage_num',
               'horse_weight','bracket','prev_date','prev_weight',
               '_rank_tmp','race_date_str'}
    feature_cols = [c for c in sub.columns
                    if c not in EXCLUDE
                    and sub[c].dtype not in [object]
                    and sub[c].notna().sum() > 0]

    X = sub[feature_cols].fillna(0)
    y = sub['is_top3']

    model = lgb.LGBMClassifier(**PARAMS[mtype])
    model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"  Train AUC: {auc:.4f}")
    print(f"  Top-10特徴量: {imp.head(10).index.tolist()}")

    joblib.dump(model, os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2.pkl'))
    joblib.dump({'feature_cols': feature_cols, 'mtype': mtype, 'auc': auc},
                os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl'))
    print(f"  Saved: lgbm_{mtype}_v2.pkl")
    return True


def main():
    print("=" * 60)
    print("  三冠馬 完全再構築 — RAW parquet 正確読み込み版")
    print("=" * 60)

    raw = load_raw_data_correctly()
    df  = preprocess(raw)

    results = {}
    for mtype in ['turf_short', 'turf_long', 'dirt']:
        results[mtype] = train_model(mtype, df)

    print(f"\n{'='*60}")
    print(f"  完了: {sum(results.values())}/3 モデル学習成功")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
