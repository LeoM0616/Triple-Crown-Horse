"""
CSVから正しいParquetを生成 → scraped_results を置き換え → 3モデル再学習
"""
import sys, os, shutil, warnings
warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')

FULL_CSV = os.path.join(DATA_DIR, 'scraped_results_full.csv')
DEST_CSV = os.path.join(DATA_DIR, 'scraped_results.csv')
DEST_PQ  = os.path.join(DATA_DIR, 'scraped_results.parquet')


# ── STEP 1: CSV → Parquet 変換 ─────────────────────────────────
def csv_to_parquet():
    print("=" * 60)
    print("STEP 1: CSV → Parquet 変換")
    df = pd.read_csv(FULL_CSV, dtype=str)  # 全列を文字列で読む（型エラー回避）
    print(f"  読込: {len(df):,}行 / {df['race_id'].nunique():,}レース")

    # Parquet保存（全列string型にしてから）
    for col in df.columns:
        df[col] = df[col].astype(str).replace('nan', '')
    df.to_parquet(DEST_PQ, index=False)

    # CSVも置き換え
    shutil.copy2(FULL_CSV, DEST_CSV)
    print(f"  ✅ {DEST_PQ}")
    print(f"  ✅ {DEST_CSV}")
    return df


# ── STEP 2: 前処理 ─────────────────────────────────────────────
def preprocess(raw_csv_df: pd.DataFrame) -> pd.DataFrame:
    print("\nSTEP 2: 前処理")
    df = raw_csv_df.copy()

    # 列名統一（JP→EN 優先）
    COL_ALIASES = {
        '着 順':'rank', 'rank':'rank_en',
        '枠 番':'bracket_jp', 'bracket':'bracket_en',
        '馬 番':'horse_num_jp', 'horse_num':'horse_num_en',
        '馬名':'horse_name_jp', 'horse_name':'horse_name_en',
        '性齢':'sex_age_jp', 'sex_age':'sex_age_en',
        '斤量':'wc_jp', 'weight_constraint':'wc_en',
        '騎手':'jockey_jp', 'jockey':'jockey_en',
        '単勝':'odds_jp', 'odds':'odds_en',
        '人 気':'pop_jp', 'popularity':'pop_en',
        '馬体重':'hw_jp', 'horse_weight':'hw_en',
        '調教師':'trainer_jp', 'trainer':'trainer_en',
    }
    df.rename(columns={c: COL_ALIASES[c] for c in df.columns if c in COL_ALIASES}, inplace=True)

    # 着順: JP列を優先
    def best(jp, en):
        return df[jp] if jp in df.columns else df.get(en, pd.Series('', index=df.index))

    df['rank']             = pd.to_numeric(best('rank','rank_en').str.extract(r'(\d+)')[0], errors='coerce')
    df['bracket']          = pd.to_numeric(best('bracket_jp','bracket_en'), errors='coerce')
    df['horse_num']        = pd.to_numeric(best('horse_num_jp','horse_num_en'), errors='coerce')
    df['horse_name']       = best('horse_name_jp','horse_name_en').astype(str).str.strip()
    df['sex_age']          = best('sex_age_jp','sex_age_en').astype(str).str.strip()
    df['weight_constraint']= pd.to_numeric(best('wc_jp','wc_en'), errors='coerce')
    df['jockey']           = best('jockey_jp','jockey_en').astype(str).str.strip()
    df['odds']             = pd.to_numeric(best('odds_jp','odds_en'), errors='coerce')
    df['popularity']       = pd.to_numeric(best('pop_jp','pop_en'), errors='coerce')
    df['horse_weight']     = best('hw_jp','hw_en').astype(str).str.strip()
    df['trainer']          = best('trainer_jp','trainer_en').astype(str).str.strip()

    df['race_id']      = df['race_id'].astype(str).str.strip()
    df['horse_id']     = df.get('horse_id', pd.Series('', index=df.index)).astype(str).str.strip()
    df['race_date']    = df.get('race_date', pd.Series('', index=df.index)).astype(str)
    df['race_info']    = df.get('race_info', pd.Series('', index=df.index)).astype(str)
    df['weather']      = df.get('weather', pd.Series('晴', index=df.index)).astype(str).replace('', '晴')
    df['track']        = df.get('track', pd.Series('良', index=df.index)).astype(str).replace('', '良')
    df['passage_rank'] = df.get('passage_rank', pd.Series('', index=df.index)).astype(str)
    df['last3f_time']  = pd.to_numeric(df.get('last3f_time', pd.Series(dtype=float)), errors='coerce')

    # 有効行のみ（着順が出ている行）
    df = df[df['rank'].notna() & (df['rank'] >= 1)].copy()
    print(f"  有効行: {len(df):,} / 年別: {df['race_id'].str[:4].value_counts().sort_index().to_dict()}")

    # 馬体重数値
    df['horse_weight_raw'] = df['horse_weight'].str.extract(r'(\d+)')[0].astype(float)

    # 時系列並び替え
    df['race_date_dt'] = pd.to_datetime(df['race_date'], errors='coerce')
    df = df.sort_values(['horse_id','race_date_dt']).reset_index(drop=True)
    df['prev_rank']     = df.groupby('horse_id')['rank'].shift(1)
    df['rest_days']     = (df['race_date_dt'] - df.groupby('horse_id')['race_date_dt'].shift(1)).dt.days
    df['weight_delta']  = (df['weight_constraint'] - df.groupby('horse_id')['weight_constraint'].shift(1)).fillna(0)
    df['weight_ratio']  = df['weight_constraint'] / (df['horse_weight_raw'] + 1e-5)

    # 上がり偏差値
    rm = df.groupby('race_id')['last3f_time'].transform('mean')
    rs = df.groupby('race_id')['last3f_time'].transform('std')
    df['relative_last3f']       = ((df['last3f_time'] - rm) / (rs + 1e-5) * 10 + 50).fillna(50)
    df['prev1_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(1).fillna(50)
    df['prev2_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(2).fillna(50)
    df['prev3_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(3).fillna(50)

    # 通過順位
    def parse_p(p):
        parts = [int(x) for x in str(p).split('-') if x.isdigit()]
        return parts[-1] if parts else 7
    df['passage_num']        = df['passage_rank'].apply(parse_p)
    df['prev1_passage_num']  = df.groupby('horse_id')['passage_num'].shift(1).fillna(7)
    df['prev2_passage_num']  = df.groupby('horse_id')['passage_num'].shift(2).fillna(7)
    df['prev3_passage_num']  = df.groupby('horse_id')['passage_num'].shift(3).fillna(7)

    # 血統
    ped_path = os.path.join(DATA_DIR, 'horse_pedigree.csv')
    if os.path.exists(ped_path):
        ped = pd.read_csv(ped_path, dtype=str)
        ped['horse_id'] = ped['horse_id'].astype(str).str.strip()
        df = df.merge(ped[['horse_id','sire','bms','lineage']], on='horse_id', how='left')
    else:
        df['sire'] = df['trainer']
        df['bms']  = df['jockey']
        df['lineage'] = 'unknown'
    df['sire']    = df['sire'].fillna('unknown').astype(str)
    df['bms']     = df['bms'].fillna('unknown').astype(str)
    df['lineage'] = df['lineage'].fillna('unknown').astype(str)
    df['sire_track_interaction'] = df['sire'] + '_' + df['track']

    df['is_long_trip'] = 0
    df['is_stay']      = 0

    # 血統エンコード
    CAT_COLS = ['sex_age','jockey','trainer','weather','track','sire','bms','lineage','sire_track_interaction']
    les = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        les[col] = le
    feat_dir = os.path.join(BASE_DIR, 'src', 'features')
    os.makedirs(feat_dir, exist_ok=True)
    joblib.dump(les, os.path.join(feat_dir, 'encoders_v3.pkl'))

    # race type
    from train_specialized import parse_race_type, assign_model_type
    parsed          = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info','')), axis=1)
    df['surface']   = [p[0] for p in parsed]
    df['distance']  = [p[1] for p in parsed]
    df['model_type']= df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['venue_code']= df['race_id'].str[4:6]
    df['year']      = df['race_id'].str[:4].astype(int)
    df['is_top3']   = (df['rank'] <= 3).astype(int)

    print(f"  model_type:\n{df['model_type'].value_counts()}")
    print(f"  is_top3: {df['is_top3'].value_counts().to_dict()}")
    return df


# ── STEP 3: 学習 ──────────────────────────────────────────────
PARAMS = {
    'turf_short': dict(n_estimators=500, learning_rate=0.02, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                       reg_alpha=0.1, reg_lambda=0.5, class_weight='balanced', random_state=42),
    'turf_long':  dict(n_estimators=600, learning_rate=0.015, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=25,
                       reg_alpha=0.2, reg_lambda=1.0, class_weight='balanced', random_state=42),
    'dirt':       dict(n_estimators=500, learning_rate=0.02, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                       reg_alpha=0.15, reg_lambda=0.8, class_weight='balanced', random_state=42),
}
EXCLUDE = {'rank','is_top3','race_id','year','surface','distance','model_type',
           'race_info','race_date','race_date_dt','odds','popularity',
           'venue_code','horse_id','horse_name','time','margin','passage_rank',
           'last3f_time','relative_last3f','passage_num','horse_weight',
           'bracket','prev_date','prev_weight',
           'rank_en','bracket_jp','bracket_en','horse_num_jp','horse_num_en',
           'horse_name_jp','horse_name_en','sex_age_jp','sex_age_en','wc_jp','wc_en',
           'jockey_jp','jockey_en','odds_jp','odds_en','pop_jp','pop_en',
           'hw_jp','hw_en','trainer_jp','trainer_en'}

def train_model(mtype: str, df: pd.DataFrame):
    from train_turf_short_final import add_jockey_venue_encoding
    from train_dirt_final import add_dirt_specific_features

    sub = df[df['model_type'] == mtype].copy()
    dist = sub['is_top3'].value_counts().to_dict()
    print(f"\n{'='*55}")
    print(f"  {mtype}: {len(sub):,}行 | is_top3: {dist}")
    if sub['is_top3'].nunique() < 2:
        print("  [SKIP] ラベルが1種類")
        return

    if mtype == 'dirt':
        tmask = pd.Series(True, index=sub.index)
        sub = add_dirt_specific_features(sub, tmask).reset_index(drop=True)

    tmask = pd.Series(True, index=sub.index)
    sub   = add_jockey_venue_encoding(sub, tmask).reset_index(drop=True)

    feat_cols = [c for c in sub.columns
                 if c not in EXCLUDE
                 and sub[c].dtype not in [object]
                 and sub[c].notna().any()
                 and pd.api.types.is_numeric_dtype(sub[c])]
    X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = sub['is_top3']

    # 時系列分割（最新20%をテスト）
    split = int(len(sub) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(**PARAMS[mtype])
    model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])

    train_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    test_auc  = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) if y_te.nunique()==2 else float('nan')

    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"  Train AUC: {train_auc:.4f}  |  Test AUC: {test_auc:.4f}")
    print(f"  Top-8: {imp.head(8).index.tolist()}")

    joblib.dump(model, os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2.pkl'))
    joblib.dump({'feature_cols': feat_cols, 'mtype': mtype, 'auc': test_auc, 'train_auc': train_auc},
                os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl'))
    print(f"  ✅ Saved: lgbm_{mtype}_v2.pkl")


def main():
    print("🏆 三冠馬 完全再構築 — 35,000件データ版")

    # STEP 1
    df_raw = csv_to_parquet()

    # STEP 2
    sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'models'))
    df = preprocess(df_raw)

    # STEP 3
    print("\nSTEP 3: 3モデル学習")
    for mtype in ['turf_short', 'turf_long', 'dirt']:
        train_model(mtype, df)

    print(f"\n{'='*55}")
    print("  🏆 完全再構築 完了！")
    print("  Streamlit アプリで「🔄 データ更新」ボタンを押してください")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
