import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import re
import joblib
COL_MAP = {
    '着 順': 'rank',
    '枠 番': 'bracket',
    '馬 番': 'horse_num',
    '馬名': 'horse_name',
    '性齢': 'sex_age',
    '斤量': 'weight_constraint',
    '騎手': 'jockey',
    'タイム': 'time',
    '着差': 'margin',
    '単勝': 'odds',
    '人 気': 'popularity',
    '馬体重': 'horse_weight',
    '調教師': 'trainer',
    'race_id': 'race_id',
    'race_info': 'race_info',
    'race_date': 'race_date',
    'horse_id': 'horse_id'
}

def preprocess_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'raw', 'scraped_results.csv')
    raw_parquet   = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    output_path   = os.path.join(base_dir, 'data', 'processed', 'model_input.csv')
    output_parquet = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')

    # ====== Raw データのキャッシュ（Parquet が存在すればそこから読む） ======
    if os.path.exists(raw_parquet):
        print(f"[CACHE HIT] Loading raw from {raw_parquet}")
        df = pd.read_parquet(raw_parquet)
    else:
        print(f"Loading raw CSV from {input_path}")
        df = pd.read_csv(input_path)
        df.rename(columns=COL_MAP, inplace=True)
        os.makedirs(os.path.dirname(raw_parquet), exist_ok=True)
        df.to_parquet(raw_parquet, index=False)
        print(f"[CACHE SAVED] {raw_parquet}")

    if 'rank' not in df.columns:
        df.rename(columns=COL_MAP, inplace=True)
    
    # 日付と数値を処理
    df['race_date_dt'] = pd.to_datetime(df['race_date'], errors='coerce')
    df['horse_id'] = df['horse_id'].astype(str)
    
    # rank（着順）を数値化（文字列が含まれる場合を除去）
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    
    # ====== 前走着順・休養日数 ======
    df = df.sort_values(by=['horse_id', 'race_date_dt'])

    df['prev_rank'] = df.groupby('horse_id')['rank'].shift(1)
    df['prev_date'] = df.groupby('horse_id')['race_date_dt'].shift(1)
    df['rest_days'] = (df['race_date_dt'] - df['prev_date']).dt.days

    # ====== 斤量の相対化 ======
    df['weight_constraint'] = pd.to_numeric(df['weight_constraint'], errors='coerce')
    df['prev_weight_constraint'] = df.groupby('horse_id')['weight_constraint'].shift(1)
    # 斤量増減（プラスは増量、マイナスは減量）
    df['weight_delta'] = df['weight_constraint'] - df['prev_weight_constraint']
    df['weight_delta'] = df['weight_delta'].fillna(0)
    # 馬体重に対する斤量負担比率（軽い馬ほど不利 → AIに学習させる）
    df['horse_weight_raw'] = df['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['weight_ratio'] = df['weight_constraint'] / (df['horse_weight_raw'] + 1e-5)
    
    # ====== 今回の追加機能②：「上がり3ハロン偏差値」「脚質・捲り」「輸送・滞在」 ======
    # 1. 上がり3ハロン偏差値
    df['last3f_time'] = pd.to_numeric(df.get('last3f_time', np.nan), errors='coerce')
    race_mean = df.groupby('race_id')['last3f_time'].transform('mean')
    race_std = df.groupby('race_id')['last3f_time'].transform('std')
    df['relative_last3f'] = (df['last3f_time'] - race_mean) / (race_std + 1e-5) * 10 + 50
    df['relative_last3f'] = df['relative_last3f'].fillna(50)  # 欠損は偏差値50
    # リーク排除：各過去走（1,2,3走前）の「上がり3ハロン偏差値」
    df['prev1_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(1).fillna(50)
    df['prev2_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(2).fillna(50)
    df['prev3_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(3).fillna(50)

    # 2. 通過順位 (過去の個別傾向)
    def parse_passage_val(p):
        if pd.isna(p) or not isinstance(p, str): return 7 # default mid-pack
        ranks = [int(v) for v in str(p).split('-') if v.isdigit()]
        if not ranks: return 7
        return ranks[-1] # last corner rank

    if 'passage_rank' in df.columns:
        df['passage_num'] = df['passage_rank'].apply(parse_passage_val)
    else:
        df['passage_num'] = 7

    # 過去3走それぞれの通過順位を保持
    df['prev1_passage_num'] = df.groupby('horse_id')['passage_num'].shift(1).fillna(7)
    df['prev2_passage_num'] = df.groupby('horse_id')['passage_num'].shift(2).fillna(7)
    df['prev3_passage_num'] = df.groupby('horse_id')['passage_num'].shift(3).fillna(7)

    # 3. 輸送距離と滞在競馬フラグ
    # race_idの上5,6桁目が競馬場コード (例: 2024[09]020411 -> 09=阪神)
    df['track_code'] = df['race_id'].astype(str).str[4:6]
    def check_transport(t_code, trainer):
        east = ['03', '04', '05', '06']
        west = ['07', '08', '09', '10']
        if '[西]' in str(trainer) and str(t_code) not in west: return 1
        if '[東]' in str(trainer) and str(t_code) not in east: return 1
        return 0
    
    df['is_long_trip'] = df.apply(lambda x: check_transport(x['track_code'], x['trainer']), axis=1)
    df['prev_track_code'] = df.groupby('horse_id')['track_code'].shift(1)
    df['is_stay'] = np.where(
        (df['track_code'].isin(['01', '02', '10'])) & (df['track_code'] == df['prev_track_code']), 1, 0
    )
    # ==============================================================
    
    # 順番を元に戻す
    df = df.sort_index()
    
    # ====== 天候と血統データの追加 ======
    # scraped_results.csv は既に weather, track を持っている想定
    
    pedigree_path = os.path.join(base_dir, 'data', 'raw', 'horse_pedigree.csv')
    if os.path.exists(pedigree_path):
        ped_df = pd.read_csv(pedigree_path, dtype={'horse_id': str})
        df = pd.merge(df, ped_df, on='horse_id', how='left')
    else:
        df['sire'] = 'unknown'
        df['bms'] = 'unknown'
        df['lineage'] = 'unknown'
        
    df['weather'] = df.get('weather', 'unknown').fillna('unknown')
    df['track'] = df.get('track', 'unknown').fillna('unknown')
    df['sire'] = df.get('sire', 'unknown').fillna('unknown')
    df['bms'] = df.get('bms', 'unknown').fillna('unknown')
    df['lineage'] = df.get('lineage', 'unknown').fillna('unknown')

    # ====== 血統 × 馬場適性 クロス特徴量 ======
    # 父（sire）系統 × 馬場状態（track: 良/稍重/重/不良）を掛け合わせた相互作用項
    # 例: 'ディープインパクト_良', 'ロードカナロア_重' → 「道悪に強い血統」をAIが学習
    df['sire_str'] = df['sire'].astype(str).fillna('unknown')
    df['track_str'] = df['track'].astype(str).fillna('unknown')
    df['sire_track_interaction'] = df['sire_str'] + '_' + df['track_str']

    # 必要な列のみ絞る
    target_cols = ['rank', 'bracket', 'horse_num', 'sex_age', 'weight_constraint', 
                   'jockey', 'odds', 'popularity', 'horse_weight', 'trainer', 'race_id',
                   'prev_rank', 'rest_days', 'weather', 'track', 'sire', 'bms', 'lineage',
                   'prev1_relative_last3f', 'prev2_relative_last3f', 'prev3_relative_last3f', 
                   'prev1_passage_num', 'prev2_passage_num', 'prev3_passage_num', 
                   'is_long_trip', 'is_stay',
                   'weight_delta', 'weight_ratio', 'sire_track_interaction']
    df = df[[c for c in target_cols if c in df.columns]].copy()
    
    # クレンジング
    df.dropna(subset=['rank'], inplace=True)
    
    df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    
    # 馬体重のクレンジング (例: '498(+4)' -> 498)
    df['horse_weight'] = df['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)
        
    # 前走データがない新馬等は平均値や特定の値で埋める
    df['prev_rank'] = df['prev_rank'].fillna(df['prev_rank'].mean())
    df['rest_days'] = df['rest_days'].fillna(90)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # カテゴリ変数の処理（ラベルエンコーディングとエンコーダの保存）
    cat_cols = ['sex_age', 'jockey', 'trainer', 'weather', 'track', 'sire', 'bms', 'lineage', 'sire_track_interaction']
    encoders = {}
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('unknown')
            le = LabelEncoder()
            # 未知の値が推論時に出た場合に備えて、'unknown' をカテゴリとして必ず含めるようになっている
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
            
    # エンコーダーの保存
    encoders_path = os.path.join(base_dir, 'src', 'features', 'encoders.pkl')
    os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
    joblib.dump(encoders, encoders_path)
    print(f"Saved LabelEncoders to {encoders_path}")

    # ====== 処理済みデータを Parquet + CSV 両方で保存 ======
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    df.to_csv(output_path, index=False)
    print(f"[CACHE SAVED] Parquet: {output_parquet}")
    print(f"Preprocessed data shape: {df.shape}")
    new_features = ['weight_delta', 'weight_ratio', 'sire_track_interaction']
    print(f"New features added: {[f for f in new_features if f in df.columns]}")
    print(df[['rank', 'prev_rank', 'rest_days']].head())

if __name__ == "__main__":
    preprocess_data()
