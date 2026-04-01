import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
import sys

# Windows環境でのUnicodeEncodeError対策
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def predict_race(race_id='202409020411'): # デフォルトは2024年大阪杯
    print(f"\n--- レース予想開始: Race ID {race_id} ---\n")
    
    # 1. 生データのロード（出馬表の代わり）
    raw_path = os.path.join(base_dir, 'data', 'raw', 'scraped_results.csv')
    df_all = pd.read_csv(raw_path, dtype={'horse_id': str, 'race_id': str})
    
    # 対象レースを抽出
    df_race = df_all[df_all['race_id'] == str(race_id)].copy()
    if df_race.empty:
        print(f"Error: Race ID {race_id} のデータが見つかりません。")
        return
        
    print(f"【レース情報】 {df_race['race_info'].iloc[0]}")
    
    # 必要な列名マッピング（preprocess.pyと同じ）
    COL_MAP = {
        '着 順': 'rank', '枠 番': 'bracket', '馬 番': 'horse_num', '馬名': 'horse_name',
        '性齢': 'sex_age', '斤量': 'weight_constraint', '騎手': 'jockey', 
        '単勝': 'odds', '人 気': 'popularity', '馬体重': 'horse_weight', 
        '調教師': 'trainer'
    }
    df_race.rename(columns=COL_MAP, inplace=True)
    
    # 2. 推論用の特徴量作成（前処理の再現）
    # 過去データを用いた前走着順等の取得
    df_race['race_date_dt'] = pd.to_datetime(df_race['race_date'], errors='coerce')
    df_all['race_date_dt'] = pd.to_datetime(df_all['race_date'], errors='coerce')
    
    prev_ranks = []
    rest_days_list = []
    
    for _, row in df_race.iterrows():
        hid = row['horse_id']
        current_date = row['race_date_dt']
        # 過去の最直近レースを探す
        past_races = df_all[(df_all['horse_id'] == hid) & (df_all['race_date_dt'] < current_date)].sort_values('race_date_dt', ascending=False)
        if not past_races.empty:
            prev_ranks.append(pd.to_numeric(past_races.iloc[0]['着 順'], errors='coerce'))
            rest_days_list.append((current_date - past_races.iloc[0]['race_date_dt']).days)
        else:
            prev_ranks.append(np.nan)
            rest_days_list.append(np.nan)
            
    df_race['prev_rank'] = prev_ranks
    df_race['rest_days'] = rest_days_list
    df_race['prev_rank'] = df_race['prev_rank'].fillna(6.9) # 訓練データの平均等で埋める
    df_race['rest_days'] = df_race['rest_days'].fillna(90)
    
    # --- 新特徴量の計算 ---
    # 1. 上がり3ハロン偏差値
    df_race['last3f_time'] = pd.to_numeric(df_race.get('last3f_time', np.nan), errors='coerce')
    race_mean = df_race['last3f_time'].mean()
    race_std = df_race['last3f_time'].std()
    df_race['relative_last3f'] = (df_race['last3f_time'] - race_mean) / (race_std + 1e-5) * 10 + 50
    df_race['relative_last3f'] = df_race['relative_last3f'].fillna(50)
    
    # 2. 脚質・捲りフラグ
    def parse_passage(p):
        if pd.isna(p) or not isinstance(p, str): return 'unknown', 0
        ranks = [int(v) for v in str(p).split('-') if v.isdigit()]
        if not ranks: return 'unknown', 0
        last_r, first_r = ranks[-1], ranks[0]
        if last_r <= 1: style = 'front'
        elif last_r <= 4: style = 'stalker'
        elif last_r <= 8: style = 'closer'
        else: style = 'rear'
        makuri = 1 if (first_r - last_r) >= 4 else 0
        return style, makuri

    if 'passage_rank' in df_race.columns:
        res = df_race['passage_rank'].apply(parse_passage)
        df_race['running_style'] = [r[0] for r in res]
        df_race['is_makuri'] = [r[1] for r in res]
    else:
        df_race['running_style'] = 'unknown'
        df_race['is_makuri'] = 0

    # 3. 輸送距離と滞在競馬フラグ
    df_race['track_code'] = df_race['race_id'].astype(str).str[4:6]
    def check_transport(t_code, trainer):
        east = ['03', '04', '05', '06']
        west = ['07', '08', '09', '10']
        if '[西]' in str(trainer) and str(t_code) not in west: return 1
        if '[東]' in str(trainer) and str(t_code) not in east: return 1
        return 0
    df_race['is_long_trip'] = df_race.apply(lambda x: check_transport(x['track_code'], x['trainer']), axis=1)
    
    # 滞在フラグ計算のため前走の開催地を取得
    prev_track_codes = []
    for _, row in df_race.iterrows():
        hid = row['horse_id']
        current_date = row['race_date_dt']
        past_races = df_all[(df_all['horse_id'] == hid) & (df_all['race_date_dt'] < current_date)].sort_values('race_date_dt', ascending=False)
        if not past_races.empty:
            prev_track_codes.append(str(past_races.iloc[0]['race_id'])[4:6])
        else:
            prev_track_codes.append('None')
            
    df_race['prev_track_code'] = prev_track_codes
    df_race['is_stay'] = np.where(
        (df_race['track_code'].isin(['01', '02', '10'])) & (df_race['track_code'] == df_race['prev_track_code']), 1, 0
    )
    # ----------------------
    
    # 血統データ結合
    pedigree_path = os.path.join(base_dir, 'data', 'raw', 'horse_pedigree.csv')
    if os.path.exists(pedigree_path):
        ped_df = pd.read_csv(pedigree_path, dtype={'horse_id': str})
        df_race = pd.merge(df_race, ped_df, on='horse_id', how='left')
        
    for c in ['weather', 'track', 'sire', 'bms', 'lineage']:
        df_race[c] = df_race.get(c, 'unknown').fillna('unknown')
        
    df_race['odds'] = pd.to_numeric(df_race['odds'], errors='coerce').fillna(20.0)
    df_race['popularity'] = pd.to_numeric(df_race['popularity'], errors='coerce').fillna(10)
    df_race['horse_weight'] = df_race['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float).fillna(470.0)
    
    # 3. エンコーダの適用
    encoders_path = os.path.join(base_dir, 'src', 'features', 'encoders.pkl')
    encoders = joblib.load(encoders_path)
    
    cat_cols = ['sex_age', 'jockey', 'trainer', 'weather', 'track', 'sire', 'bms', 'lineage', 'running_style']
    for c in cat_cols:
        if c in df_race.columns:
            le = encoders.get(c)
            df_race[c] = df_race[c].astype(str).fillna('unknown')
            if le:
                # 未知データが含まれた場合は 'unknown' クラスにマッピングする処理
                classes = list(le.classes_)
                df_race[c] = df_race[c].apply(lambda x: x if x in classes else 'unknown')
                df_race[c] = le.transform(df_race[c])
            else:
                df_race[c] = 0
                
    # 4. 予測の実行
    model_path = os.path.join(base_dir, 'src', 'models', 'lgbm_model.pkl')
    model = joblib.load(model_path)
    
    feature_cols = ['bracket', 'horse_num', 'sex_age', 'weight_constraint', 
                    'jockey', 'odds', 'popularity', 'horse_weight', 'trainer', 
                    'prev_rank', 'rest_days', 'weather', 'track', 'sire', 'bms', 'lineage',
                    'relative_last3f', 'running_style', 'is_makuri', 'is_long_trip', 'is_stay']
    
    # 欠損値等を埋める
    X = df_race[feature_cols].copy()
    X.fillna(X.mean(numeric_only=True), inplace=True)
    X.fillna(0, inplace=True)
    
    preds = model.predict(X)
    df_race['pred_score'] = preds
    
    # 順位付け (スコアが小さいほど1着に近い)
    df_race['pred_rank'] = df_race['pred_score'].rank(method='min')
    
    # 結果の表示用フォーマット
    result_df = pd.DataFrame({
        '馬番': df_race['horse_num'],
        '馬名': df_race['horse_name'],
        'AI予想着順': df_race['pred_rank'].astype(int),
        'スコア(予測値)': df_race['pred_score'].round(2),
        '実際の着順': df_race['rank']
    })
    
    result_df = result_df.sort_values('AI予想着順')
    
    print("\n【AI着順予想ランキング（答え合わせ）】")
    print("-" * 65)
    print(result_df.to_string(index=False))
    print("-" * 65)
    print("\n AIが『馬券に絡む』と判断したトップ3頭:")
    for i in range(3):
        row = result_df.iloc[i]
        print(f"  {i+1}位指名! 馬番{row['馬番']} {row['馬名']}  (実際の結果: {row['実際の着順']})")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else '202409020411'
    predict_race(target)
