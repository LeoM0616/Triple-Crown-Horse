"""
三冠馬 v3 - RAW parquetから直接前処理してモデル学習
現在利用可能な2941行（2023-2026年Q1）で学習・予測システムを構築する。
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

CAT_COLS = ['sex_age', 'jockey', 'trainer', 'weather', 'track', 'sire', 'bms',
            'lineage', 'sire_track_interaction']

PARAMS = {
    'turf_short': dict(n_estimators=300, learning_rate=0.03, num_leaves=31,
                       max_depth=5, min_child_samples=5, class_weight='balanced', random_state=42),
    'turf_long':  dict(n_estimators=400, learning_rate=0.025, num_leaves=31,
                       max_depth=5, min_child_samples=8, class_weight='balanced', random_state=42),
    'dirt':       dict(n_estimators=400, learning_rate=0.025, num_leaves=31,
                       max_depth=5, min_child_samples=8, class_weight='balanced', random_state=42),
}


def preprocess_raw() -> pd.DataFrame:
    """RAW parquetからフル前処理してDataFrameを返す"""
    raw_pq = os.path.join(BASE_DIR, 'data', 'raw', 'scraped_results.parquet')
    raw = pd.read_parquet(raw_pq)

    # 列名 → 統一名マッピング（JP版と英語版の両方を処理）
    COL_ALIASES = {
        '着 順': 'rank',     'rank': 'rank',
        '枠 番': 'bracket',  'bracket': 'bracket',
        '馬 番': 'horse_num','horse_num': 'horse_num',
        '馬名': 'horse_name','horse_name': 'horse_name',
        '性齢': 'sex_age',   'sex_age': 'sex_age',
        '斤量': 'weight_constraint', 'weight_constraint': 'weight_constraint',
        '騎手': 'jockey',    'jockey': 'jockey',
        'タイム': 'time',    'time': 'time',
        '着差': 'margin',    'margin': 'margin',
        '単勝': 'odds',      'odds': 'odds',
        '人 気': 'popularity','popularity': 'popularity',
        '馬体重': 'horse_weight','horse_weight': 'horse_weight',
        '調教師': 'trainer', 'trainer': 'trainer',
        'race_id': 'race_id','race_info': 'race_info','race_date': 'race_date',
        'horse_id': 'horse_id','weather': 'weather','track': 'track',
        'passage_rank': 'passage_rank','last3f_time': 'last3f_time',
    }
    # 各ターゲット列名ごとに最初に見つかった列だけ残す
    used_targets = set()
    cols_to_keep = {}
    for col in raw.columns:
        target = COL_ALIASES.get(col)
        if target and target not in used_targets:
            cols_to_keep[col] = target
            used_targets.add(target)

    raw = raw[[c for c in raw.columns if c in cols_to_keep]].rename(columns=cols_to_keep)

    raw['race_id']  = raw['race_id'].astype(str)
    raw['horse_id'] = raw['horse_id'].astype(str)
    raw['race_date_dt'] = pd.to_datetime(raw['race_date'], errors='coerce')
    raw['rank'] = raw['着 順'].astype(str).str.extract(r'(\d+)')[0] if '着 順' in raw.columns else \
                  pd.to_numeric(raw.get('rank', pd.Series(dtype=float)), errors='coerce')
    raw['rank'] = pd.to_numeric(raw['rank'], errors='coerce')

    # 基本特徴量
    raw['weight_constraint'] = pd.to_numeric(raw['weight_constraint'], errors='coerce')
    raw['horse_weight_raw']  = raw['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    raw['odds']       = pd.to_numeric(raw['odds'], errors='coerce')
    raw['popularity'] = pd.to_numeric(raw['popularity'], errors='coerce')
    raw['horse_num']  = pd.to_numeric(raw['horse_num'], errors='coerce')

    # 並べ替えて時系列特徴量
    raw = raw.sort_values(['horse_id', 'race_date_dt']).reset_index(drop=True)
    raw['prev_rank']  = raw.groupby('horse_id')['rank'].shift(1)
    raw['prev_date']  = raw.groupby('horse_id')['race_date_dt'].shift(1)
    raw['rest_days']  = (raw['race_date_dt'] - raw['prev_date']).dt.days

    raw['prev_weight'] = raw.groupby('horse_id')['weight_constraint'].shift(1)
    raw['weight_delta'] = raw['weight_constraint'] - raw['prev_weight']
    raw['weight_delta'] = raw['weight_delta'].fillna(0)
    raw['weight_ratio'] = raw['weight_constraint'] / (raw['horse_weight_raw'] + 1e-5)

    # 上がり偏差値
    raw['last3f_time'] = pd.to_numeric(raw['last3f_time'], errors='coerce')
    r_mean = raw.groupby('race_id')['last3f_time'].transform('mean')
    r_std  = raw.groupby('race_id')['last3f_time'].transform('std')
    raw['relative_last3f'] = ((raw['last3f_time'] - r_mean) / (r_std + 1e-5) * 10 + 50).fillna(50)
    raw['prev1_relative_last3f'] = raw.groupby('horse_id')['relative_last3f'].shift(1).fillna(50)
    raw['prev2_relative_last3f'] = raw.groupby('horse_id')['relative_last3f'].shift(2).fillna(50)
    raw['prev3_relative_last3f'] = raw.groupby('horse_id')['relative_last3f'].shift(3).fillna(50)

    # 通過順位
    def parse_passage(p):
        if pd.isna(p) or not isinstance(p, str): return 7
        parts = [int(v) for v in str(p).split('-') if v.isdigit()]
        return parts[-1] if parts else 7
    if 'passage_rank' in raw.columns:
        raw['passage_num'] = raw['passage_rank'].apply(parse_passage)
    else:
        raw['passage_num'] = 7
    raw['prev1_passage_num'] = raw.groupby('horse_id')['passage_num'].shift(1).fillna(7)
    raw['prev2_passage_num'] = raw.groupby('horse_id')['passage_num'].shift(2).fillna(7)
    raw['prev3_passage_num'] = raw.groupby('horse_id')['passage_num'].shift(3).fillna(7)

    # 馬場状態・天候 クリーニング
    raw['weather'] = raw['weather'].fillna('晴')
    raw['track']   = raw['track'].fillna('良')

    # 血統情報（horse_pedigree.csvから）
    ped_path = os.path.join(BASE_DIR, 'data', 'raw', 'horse_pedigree.csv')
    if os.path.exists(ped_path):
        ped = pd.read_csv(ped_path, dtype=str)
        ped['horse_id'] = ped['horse_id'].astype(str)
        raw = raw.merge(ped[['horse_id','sire','bms','lineage']], on='horse_id', how='left')
    else:
        raw['sire'] = raw['trainer']   # fallback
        raw['bms']  = raw['jockey']
        raw['lineage'] = 'unknown'

    raw['sire']    = raw['sire'].fillna('unknown')
    raw['bms']     = raw['bms'].fillna('unknown')
    raw['lineage'] = raw['lineage'].fillna('unknown')
    raw['sire_track_interaction'] = raw['sire'].astype(str) + '_' + raw['track'].astype(str)

    # 輸送フラグ
    raw['is_long_trip'] = 0
    raw['is_stay'] = 0

    # カテゴリエンコード
    les = {}
    for col in CAT_COLS:
        if col in raw.columns:
            le = LabelEncoder()
            raw[col] = le.fit_transform(raw[col].astype(str))
            les[col] = le

    # race type付与
    parsed = raw.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    raw['surface']    = [p[0] for p in parsed]
    raw['distance']   = [p[1] for p in parsed]
    raw['model_type'] = raw.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    raw['venue_code'] = raw['race_id'].str[4:6]
    raw['year'] = raw['race_id'].str[:4].astype(int)
    raw['is_top3'] = (raw['rank'] <= 3).astype(int)

    joblib.dump(les, os.path.join(BASE_DIR, 'src', 'features', 'encoders_v3.pkl'))
    return raw


def train_model(mtype: str, df: pd.DataFrame):
    sub = df[df['model_type'] == mtype].copy()
    print(f"\n{'='*55}")
    print(f"  {mtype}: {len(sub)} rows, is_top3 dist: {sub['is_top3'].value_counts().to_dict()}")

    if sub['is_top3'].nunique() < 2:
        print("  [SKIP] ラベルが1種類")
        return

    if mtype == 'dirt':
        tmask = pd.Series([True]*len(sub), index=sub.index)
        sub = add_dirt_specific_features(sub, tmask)
        sub = sub.reset_index(drop=True)

    tmask = pd.Series([True]*len(sub), index=sub.index)
    sub   = add_jockey_venue_encoding(sub, tmask)
    sub   = sub.reset_index(drop=True)

    EXCLUDE = set(['rank','is_top3','race_id','year','surface','distance','model_type',
                   'race_info','race_date','race_date_dt','odds','popularity',
                   'venue_code','horse_id','horse_name','time','margin','passage_rank',
                   'last3f_time','relative_last3f','passage_num',
                   'horse_weight','bracket','prev_date','prev_weight'])
    feature_cols = [c for c in sub.columns if c not in EXCLUDE and sub[c].dtype != object]
    feature_cols = [c for c in feature_cols if sub[c].notna().sum() > 0]

    X = sub[feature_cols].fillna(0)
    y = sub['is_top3']

    model = lgb.LGBMClassifier(**PARAMS[mtype])
    model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"  Train AUC: {auc:.4f}")
    print(f"  Top-8: {imp.head(8).index.tolist()}")

    joblib.dump(model, os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2.pkl'))
    joblib.dump({'feature_cols': feature_cols, 'mtype': mtype, 'auc': auc},
                os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl'))
    print(f"  Saved: lgbm_{mtype}_v2.pkl")


def main():
    print("前処理実行中...")
    df = preprocess_raw()
    print(f"Total rows: {len(df)}")
    print(f"model_type:\n{df['model_type'].value_counts()}")
    print(f"is_top3:\n{df['is_top3'].value_counts()}")

    for mtype in ['turf_short', 'turf_long', 'dirt']:
        train_model(mtype, df)

    print(f"\n三冠馬 v3 — 全モデル再学習完了!")


if __name__ == '__main__':
    main()
