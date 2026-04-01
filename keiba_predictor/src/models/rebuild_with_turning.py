"""
回り順適性特徴量を加えた 35,000件フル再学習
VENUE_TURN で左/右を判別 → 馬ごとの左右別偏差値を算出
"""
import sys, os, warnings
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
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')
FEAT_DIR  = os.path.join(BASE_DIR, 'src', 'features')

# ─── 競馬場 → 回り順 ─────────────────────────────────────────
# JRA 10場: 左=L 右=R
VENUE_TURN = {
    '01': 'L',  # 札幌（左回り）
    '02': 'R',  # 函館（右回り）
    '03': 'R',  # 福島（右回り）
    '04': 'L',  # 新潟（左回り）
    '05': 'L',  # 東京（左回り）
    '06': 'R',  # 中山（右回り）
    '07': 'L',  # 中京（左回り）
    '08': 'R',  # 京都（右回り）
    '09': 'R',  # 阪神（右回り）
    '10': 'R',  # 小倉（右回り）
}

PARAMS = {
    'turf_short': dict(n_estimators=600, learning_rate=0.02, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                       reg_alpha=0.1, reg_lambda=0.5, class_weight='balanced', random_state=42),
    'turf_long':  dict(n_estimators=700, learning_rate=0.015, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=25,
                       reg_alpha=0.2, reg_lambda=1.0, class_weight='balanced', random_state=42),
    'dirt':       dict(n_estimators=600, learning_rate=0.02, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                       reg_alpha=0.15, reg_lambda=0.8, class_weight='balanced', random_state=42),
}

EXCLUDE_BASE = {
    'rank','is_top3','race_id','year','surface','distance','model_type',
    'race_info','race_date','race_date_dt','odds','popularity','venue_code',
    'horse_id','horse_name','time','margin','passage_rank','last3f_time',
    'relative_last3f','passage_num','horse_weight','bracket','prev_date','prev_weight',
    # 日本語重複列
    'rank_en','bracket_jp','bracket_en','horse_num_jp','horse_num_en',
    'horse_name_jp','horse_name_en','sex_age_jp','sex_age_en','wc_jp','wc_en',
    'jockey_jp','jockey_en','odds_jp','odds_en','pop_jp','pop_en',
    'hw_jp','hw_en','trainer_jp','trainer_en',
}


def add_turning_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    馬ごとの「左回り・右回り別 偏差値」を算出して追加
    ※ グループ内での過去成績集計（shift ベース）でリーク防止
    """
    df = df.copy()
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df['turn_dir']   = df['venue_code'].map(VENUE_TURN).fillna('R')
    df['is_left']    = (df['turn_dir'] == 'L').astype(int)

    # 時系列ソート済み前提（preprocess で sort 済み）
    # horse_id × turn_dir 別に過去の平均着順を rolling で計算
    df = df.sort_values(['horse_id', 'race_date_dt']).reset_index(drop=True)

    # 過去 n 戦の左/右別 平均着順（expanding, shift して現在レースを除外）
    for turn, col_sfx in [('L', 'left'), ('R', 'right')]:
        mask = df['turn_dir'] == turn
        df[f'turn_{col_sfx}_rank_avg'] = np.nan
        grp = df[mask].groupby('horse_id')['rank']
        expanding_mean = grp.transform(lambda x: x.shift(1).expanding().mean())
        df.loc[mask, f'turn_{col_sfx}_rank_avg'] = expanding_mean

    # NaN → overall_avg で補完
    overall_avg = df.groupby('horse_id')['rank'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(df['rank'].mean())

    df['turn_left_rank_avg']  = df['turn_left_rank_avg'].fillna(overall_avg)
    df['turn_right_rank_avg'] = df['turn_right_rank_avg'].fillna(overall_avg)

    # 偏差スコア（正 = その回りで他の場所より好走）
    df['turn_left_deviation']  = overall_avg - df['turn_left_rank_avg']
    df['turn_right_deviation'] = overall_avg - df['turn_right_rank_avg']
    # 左右どちらが得意か（正 = 左得意）
    df['turn_preference'] = df['turn_left_deviation'] - df['turn_right_deviation']
    # 今回コースとの適性マッチスコア
    df['turn_match_score'] = np.where(
        df['is_left'] == 1, df['turn_left_deviation'], df['turn_right_deviation']
    )

    return df


def load_and_preprocess() -> pd.DataFrame:
    print("STEP 1: 35k CSVロード & 前処理")
    full_csv = os.path.join(DATA_DIR, 'scraped_results_full.csv')
    base_csv = os.path.join(DATA_DIR, 'scraped_results.csv')
    src = full_csv if os.path.exists(full_csv) else base_csv

    df_raw = pd.read_csv(src, dtype=str)
    print(f"  読込: {len(df_raw):,}行 / {df_raw['race_id'].nunique():,}レース")

    # JP/EN 列の正規化
    COL_ALIASES = {
        '着 順':'rank','枠 番':'bracket','馬 番':'horse_num','馬名':'horse_name',
        '性齢':'sex_age','斤量':'weight_constraint','騎手':'jockey','タイム':'time',
        '着差':'margin','単勝':'odds','人 気':'popularity','馬体重':'horse_weight',
        '調教師':'trainer',
    }
    # JP列がなければEN列をそのまま使う
    for jp, en in COL_ALIASES.items():
        if jp in df_raw.columns:
            df_raw[en] = df_raw[jp]

    df_raw['race_id']      = df_raw['race_id'].astype(str).str.strip()
    df_raw['horse_id']     = df_raw.get('horse_id', pd.Series('', index=df_raw.index)).astype(str).str.strip()
    df_raw['race_date']    = df_raw.get('race_date', pd.Series('', index=df_raw.index)).astype(str)
    df_raw['race_info']    = df_raw.get('race_info', pd.Series('', index=df_raw.index)).astype(str)
    df_raw['weather']      = df_raw.get('weather', pd.Series('晴', index=df_raw.index)).astype(str).replace('', '晴')
    df_raw['track']        = df_raw.get('track', pd.Series('良', index=df_raw.index)).astype(str).replace('', '良')
    df_raw['passage_rank'] = df_raw.get('passage_rank', pd.Series('', index=df_raw.index)).astype(str)
    df_raw['last3f_time']  = pd.to_numeric(df_raw.get('last3f_time', pd.Series(dtype=float)), errors='coerce')

    for col in ['rank','bracket','horse_num','weight_constraint','odds','popularity']:
        df_raw[col] = pd.to_numeric(df_raw[col].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

    df_raw['horse_weight_raw'] = df_raw['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)

    # 有効行
    df_raw = df_raw[df_raw['rank'].notna() & (df_raw['rank'] >= 1)].copy()
    df_raw['race_date_dt'] = pd.to_datetime(df_raw['race_date'], errors='coerce')
    df_raw = df_raw.sort_values(['horse_id', 'race_date_dt']).reset_index(drop=True)

    # 時系列特徴量
    df_raw['prev_rank']     = df_raw.groupby('horse_id')['rank'].shift(1)
    df_raw['rest_days']     = (df_raw['race_date_dt'] - df_raw.groupby('horse_id')['race_date_dt'].shift(1)).dt.days
    df_raw['weight_delta']  = (df_raw['weight_constraint'] - df_raw.groupby('horse_id')['weight_constraint'].shift(1)).fillna(0)
    df_raw['weight_ratio']  = df_raw['weight_constraint'] / (df_raw['horse_weight_raw'] + 1e-5)

    # 上がり偏差値
    rm = df_raw.groupby('race_id')['last3f_time'].transform('mean')
    rs = df_raw.groupby('race_id')['last3f_time'].transform('std')
    df_raw['relative_last3f']       = ((df_raw['last3f_time'] - rm) / (rs + 1e-5) * 10 + 50).fillna(50)
    df_raw['prev1_relative_last3f'] = df_raw.groupby('horse_id')['relative_last3f'].shift(1).fillna(50)
    df_raw['prev2_relative_last3f'] = df_raw.groupby('horse_id')['relative_last3f'].shift(2).fillna(50)
    df_raw['prev3_relative_last3f'] = df_raw.groupby('horse_id')['relative_last3f'].shift(3).fillna(50)

    # 通過順位
    def parse_p(p):
        parts = [int(x) for x in str(p).split('-') if x.isdigit()]
        return parts[-1] if parts else 7
    df_raw['passage_num']        = df_raw['passage_rank'].apply(parse_p)
    df_raw['prev1_passage_num']  = df_raw.groupby('horse_id')['passage_num'].shift(1).fillna(7)
    df_raw['prev2_passage_num']  = df_raw.groupby('horse_id')['passage_num'].shift(2).fillna(7)
    df_raw['prev3_passage_num']  = df_raw.groupby('horse_id')['passage_num'].shift(3).fillna(7)

    # 血統
    ped_path = os.path.join(DATA_DIR, 'horse_pedigree.csv')
    if os.path.exists(ped_path):
        ped = pd.read_csv(ped_path, dtype=str)
        ped['horse_id'] = ped['horse_id'].astype(str).str.strip()
        df_raw = df_raw.merge(ped[['horse_id','sire','bms','lineage']], on='horse_id', how='left')
    else:
        df_raw['sire']    = df_raw.get('trainer', 'unknown')
        df_raw['bms']     = df_raw.get('jockey', 'unknown')
        df_raw['lineage'] = 'unknown'
    for c in ['sire','bms','lineage']:
        df_raw[c] = df_raw[c].fillna('unknown').astype(str)
    df_raw['sire_track_interaction'] = df_raw['sire'] + '_' + df_raw['track']
    df_raw['is_long_trip'] = 0
    df_raw['is_stay']      = 0

    # ── 回り順適性特徴量（NEW） ──────────────────────────────────
    print("  回り順適性特徴量を算出中...")
    df_raw = add_turning_features(df_raw)

    # カテゴリエンコード
    CAT_COLS = ['sex_age','jockey','trainer','weather','track','sire','bms','lineage','sire_track_interaction']
    les = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df_raw[col] = le.fit_transform(df_raw[col].fillna('unknown').astype(str))
        les[col] = le
    os.makedirs(FEAT_DIR, exist_ok=True)
    joblib.dump(les, os.path.join(FEAT_DIR, 'encoders_v3.pkl'))

    # Race type
    parsed           = df_raw.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info','')), axis=1)
    df_raw['surface']    = [p[0] for p in parsed]
    df_raw['distance']   = [p[1] for p in parsed]
    df_raw['model_type'] = df_raw.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df_raw['year']       = df_raw['race_id'].str[:4].astype(int)
    df_raw['is_top3']    = (df_raw['rank'] <= 3).astype(int)

    # 馬ごとの回り順統計をLUT として保存（推論時に使用）
    turn_lut = df_raw.groupby('horse_id').agg(
        turn_left_rank_avg  = ('turn_left_rank_avg', 'last'),
        turn_right_rank_avg = ('turn_right_rank_avg', 'last'),
        turn_left_deviation = ('turn_left_deviation', 'last'),
        turn_right_deviation= ('turn_right_deviation', 'last'),
        turn_preference     = ('turn_preference', 'last'),
    ).reset_index()
    joblib.dump(turn_lut, os.path.join(FEAT_DIR, 'turning_lut.pkl'))
    print(f"  回り順LUT saved: {len(turn_lut):,}頭")

    year_dist = df_raw['race_id'].str[:4].value_counts().sort_index().to_dict()
    mt_dist   = df_raw['model_type'].value_counts().to_dict()
    t3_dist   = df_raw['is_top3'].value_counts().to_dict()
    print(f"  年別: {year_dist}")
    print(f"  model_type: {mt_dist}")
    print(f"  is_top3: {t3_dist}")
    return df_raw


def train_model(mtype: str, df: pd.DataFrame):
    sub  = df[df['model_type'] == mtype].copy()
    dist = sub['is_top3'].value_counts().to_dict()
    print(f"\n{'='*58}")
    print(f"  {'▶ '+mtype.upper():<20} {len(sub):,}行 | is_top3: {dist}")
    if sub['is_top3'].nunique() < 2:
        print("  [SKIP] ラベルが1種類")
        return None

    if mtype == 'dirt':
        tmask = pd.Series(True, index=sub.index)
        sub = add_dirt_specific_features(sub, tmask).reset_index(drop=True)

    tmask = pd.Series(True, index=sub.index)
    sub   = add_jockey_venue_encoding(sub, tmask).reset_index(drop=True)

    feat_cols = [
        c for c in sub.columns
        if c not in EXCLUDE_BASE
        and pd.api.types.is_numeric_dtype(sub[c])
        and sub[c].notna().any()
    ]
    X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = sub['is_top3']

    # 時系列分割: 2026年をテスト
    split_mask = sub['year'] < 2026
    X_tr, y_tr = X[split_mask], y[split_mask]
    X_te, y_te = X[~split_mask], y[~split_mask]

    model = lgb.LGBMClassifier(**PARAMS[mtype])
    model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])

    tr_auc = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    te_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) if y_te.nunique()==2 else float('nan')

    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    turn_feats = [f for f in imp.index[:20] if 'turn' in f or 'left' in f or 'right' in f]
    print(f"  Train AUC: {tr_auc:.4f}  Test AUC: {te_auc:.4f}")
    print(f"  Top-10: {imp.head(10).index.tolist()}")
    print(f"  回り順特徴量ランク: {turn_feats}")

    joblib.dump(model, os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2.pkl'))
    joblib.dump({'feature_cols': feat_cols, 'mtype': mtype,
                 'auc': te_auc, 'train_auc': tr_auc,
                 'top_features': imp.head(20).to_dict()},
                os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl'))
    print(f"  ✅ Saved: lgbm_{mtype}_v2.pkl")
    return imp


def main():
    print("🏆 三冠馬 最強版再構築 — 回り順適性追加")
    print("=" * 58)
    df = load_and_preprocess()
    sys.path.insert(0, MODEL_DIR)

    for mtype in ['turf_short', 'turf_long', 'dirt']:
        train_model(mtype, df)

    print(f"\n{'='*58}")
    print("  🏆 完了！ Streamlitアプリで「🔄 更新」を押してください")
    print(f"{'='*58}")


if __name__ == '__main__':
    main()
