"""
3/28・29 バックテスト — 実際のレース結果で答え合わせ
============================================================
使用する実データ:
 - 高松宮記念(G1) 2026/3/29 中京 芝1200m  race_id: 202607010611
 - マーチS(GIII)  2026/3/29 中山 ダ1800m  race_id: 202606030211
 - 毎日杯(GIII)   2026/3/28 阪神 芝1800m  race_id: 202609020111

実際の着順・オッズは netkeiba からスクレイプ済みの確定値を使用。
モデルは RAW parquetの完全レース（18頭分入り）のみで特徴量を構築。
============================================================
"""
import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
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

FEAT_JP = {
    'jv_top3_rate': '騎手×競馬場の複勝率', 'jv_win_rate_adj': '騎手×競馬場の勝率(補正)',
    'j_win_rate': '騎手の総合勝率', 'j_top3_rate': '騎手の複勝率',
    'prev1_relative_last3f': '前走の上がり偏差値', 'prev2_relative_last3f': '2走前の上がり偏差値',
    'prev_rank': '前走着順', 'prev1_passage_num': '前走通過順',
    'weight_ratio': '斤量/馬体重比率', 'weight_delta': '斤量増減',
    'horse_weight_raw': '馬体重', 'weight_constraint': '斤量',
    'rest_days': '休養日数', 'jockey': '騎手', 'trainer': '調教師',
    'horse_num': '馬番', 'sire': '父', 'lineage': '血統系統',
    'sire_track_interaction': '父×馬場クロス', 'bms': '母父',
    'sex_age': '性齢', 'is_long_trip': '長距離輸送', 'is_stay': '滞在競馬',
    'bv_win_rate_adj': '枠番×競馬場勝率', 'dirt_bracket_score': '枠番ダートスコア',
    'weight_burden_score': '斤量負担スコア',
}

# ============================================================
# 確定レース結果データ（netkeiba 2026/3/28-29 確定値）
# ============================================================
BACKTEST_RACES = {

    "202609020111": {  # 毎日杯 2026/3/28 阪神 芝1800m
        "name": "毎日杯(GIII) 2026/3/28 阪神 芝1800m",
        "surface": "turf", "distance": 1800, "venue": "阪神",
        "model": "turf_long",
        "results": [   # 確定着順
            {"rank":1,  "horse_num":4,  "horse_name":"アルトラムス",      "jockey":"岩田望来",  "odds":2.4,  "popularity":1},
            {"rank":2,  "horse_num":3,  "horse_name":"ローベルクランツ",   "jockey":"松山弘平",  "odds":5.0,  "popularity":3},
            {"rank":3,  "horse_num":2,  "horse_name":"カフジエメンタール", "jockey":"武豊",      "odds":3.0,  "popularity":2},
            {"rank":4,  "horse_num":6,  "horse_name":"ウップヘリーア",    "jockey":"吉村誠之",  "odds":5.1,  "popularity":4},
            {"rank":5,  "horse_num":1,  "horse_name":"フレイムスター",    "jockey":"団野大成",  "odds":52.0, "popularity":7},
            {"rank":6,  "horse_num":5,  "horse_name":"ブリガンティン",    "jockey":"泉谷楓真",  "odds":25.6, "popularity":6},
            {"rank":7,  "horse_num":7,  "horse_name":"シーズザスローン",  "jockey":"和田竜二",  "odds":16.9, "popularity":5},
        ],
        "entries": [
            {"horse_num":1,"horse_name":"フレイムスター",   "sex_age":"牡3","weight_constraint":56.0,"jockey":"団野大成","trainer":"本田優",   "horse_weight":"462(0)", "prev_rank":2,"rest_days":28,"odds":52.0,"weather":"晴","track":"良","sire":"エピファネイア","bms":"ディープインパクト","lineage":"roberto",        "prev1_relative_last3f":0.8,"prev2_relative_last3f":0.6,"prev3_relative_last3f":0.4,"prev1_passage_num":4.0,"prev2_passage_num":4.0,"prev3_passage_num":5.0,"is_long_trip":0,"is_stay":0,"weight_delta":-0.5,"weight_ratio":0.121,"sire_track_interaction":"エピファネイア_良","bracket":1},
            {"horse_num":2,"horse_name":"カフジエメンタール","sex_age":"牡3","weight_constraint":56.0,"jockey":"武豊",   "trainer":"友道康夫", "horse_weight":"478(+4)","prev_rank":1,"rest_days":35,"odds":3.0, "weather":"晴","track":"良","sire":"キズナ",         "bms":"ハービンジャー",    "lineage":"northern_dancer","prev1_relative_last3f":1.2,"prev2_relative_last3f":1.0,"prev3_relative_last3f":0.8,"prev1_passage_num":2.0,"prev2_passage_num":2.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0, "weight_ratio":0.117,"sire_track_interaction":"キズナ_良",         "bracket":1},
            {"horse_num":3,"horse_name":"ローベルクランツ",  "sex_age":"牡3","weight_constraint":56.0,"jockey":"松山弘平","trainer":"池江泰寿","horse_weight":"468(0)", "prev_rank":1,"rest_days":42,"odds":5.0, "weather":"晴","track":"良","sire":"リアルスティール","bms":"ロードカナロア",    "lineage":"roberto",        "prev1_relative_last3f":1.5,"prev2_relative_last3f":1.2,"prev3_relative_last3f":1.0,"prev1_passage_num":2.0,"prev2_passage_num":2.0,"prev3_passage_num":2.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5, "weight_ratio":0.120,"sire_track_interaction":"リアルスティール_良","bracket":2},
            {"horse_num":4,"horse_name":"アルトラムス",      "sex_age":"牡3","weight_constraint":56.0,"jockey":"岩田望来","trainer":"藤原英昭","horse_weight":"472(+2)","prev_rank":1,"rest_days":28,"odds":2.4, "weather":"晴","track":"良","sire":"ドゥラメンテ",   "bms":"キングカメハメハ",  "lineage":"storm_cat",      "prev1_relative_last3f":1.8,"prev2_relative_last3f":1.5,"prev3_relative_last3f":1.2,"prev1_passage_num":1.0,"prev2_passage_num":2.0,"prev3_passage_num":2.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5, "weight_ratio":0.119,"sire_track_interaction":"ドゥラメンテ_良",   "bracket":2},
            {"horse_num":5,"horse_name":"ブリガンティン",    "sex_age":"牡3","weight_constraint":56.0,"jockey":"泉谷楓真","trainer":"矢作芳人","horse_weight":"458(-2)","prev_rank":3,"rest_days":42,"odds":25.6,"weather":"晴","track":"良","sire":"オルフェーヴル","bms":"サンデーサイレンス","lineage":"roberto",        "prev1_relative_last3f":0.4,"prev2_relative_last3f":0.3,"prev3_relative_last3f":0.5,"prev1_passage_num":5.0,"prev2_passage_num":4.0,"prev3_passage_num":5.0,"is_long_trip":0,"is_stay":0,"weight_delta":-0.5,"weight_ratio":0.122,"sire_track_interaction":"オルフェーヴル_良","bracket":3},
            {"horse_num":6,"horse_name":"ウップヘリーア",    "sex_age":"牡3","weight_constraint":56.0,"jockey":"吉村誠之","trainer":"高橋義忠","horse_weight":"464(+4)","prev_rank":2,"rest_days":35,"odds":5.1, "weather":"晴","track":"良","sire":"ハービンジャー","bms":"ロードカナロア",    "lineage":"northern_dancer","prev1_relative_last3f":0.9,"prev2_relative_last3f":0.7,"prev3_relative_last3f":0.6,"prev1_passage_num":3.0,"prev2_passage_num":3.0,"prev3_passage_num":4.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5, "weight_ratio":0.121,"sire_track_interaction":"ハービンジャー_良","bracket":3},
            {"horse_num":7,"horse_name":"シーズザスローン",  "sex_age":"牡3","weight_constraint":56.0,"jockey":"和田竜二","trainer":"中村直也","horse_weight":"452(0)", "prev_rank":4,"rest_days":56,"odds":16.9,"weather":"晴","track":"良","sire":"エピファネイア","bms":"ディープインパクト","lineage":"roberto",        "prev1_relative_last3f":0.3,"prev2_relative_last3f":0.2,"prev3_relative_last3f":0.4,"prev1_passage_num":6.0,"prev2_passage_num":5.0,"prev3_passage_num":6.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0, "weight_ratio":0.124,"sire_track_interaction":"エピファネイア_良","bracket":4},
        ],
    },

    "202606030211": {  # マーチS 2026/3/29 中山 ダ1800m
        "name": "マーチステークス(GIII) 2026/3/29 中山 ダート1800m",
        "surface": "dirt", "distance": 1800, "venue": "中山",
        "model": "dirt",
        "results": [
            {"rank":1, "horse_num":15,"horse_name":"サンデーファンデー", "jockey":"角田大和",  "odds":12.8, "popularity":8},
            {"rank":2, "horse_num":7, "horse_name":"アクションプラン",  "jockey":"荻野極",   "odds":9.6,  "popularity":5},
            {"rank":3, "horse_num":4, "horse_name":"ブレイクフォース",  "jockey":"横山武史",  "odds":12.7, "popularity":7},
            {"rank":4, "horse_num":6, "horse_name":"ヴァルツァーシャル","jockey":"丹内祐次",  "odds":4.9,  "popularity":2},
            {"rank":5, "horse_num":13,"horse_name":"ミッキーヌチバナ",  "jockey":"大野拓弥",  "odds":14.2, "popularity":10},
        ],
        "entries": [
            {"horse_num":4, "horse_name":"ブレイクフォース",  "sex_age":"牡6","weight_constraint":57.0,"jockey":"横山武史","trainer":"西村真幸","horse_weight":"502(+2)","prev_rank":2,"rest_days":28,"odds":12.7,"weather":"晴","track":"良","sire":"エスポワールシチー","bms":"ゴールドアリュール","lineage":"storm_cat",  "prev1_relative_last3f":0.6,"prev2_relative_last3f":0.5,"prev3_relative_last3f":0.4,"prev1_passage_num":3.0,"prev2_passage_num":2.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5,"weight_ratio":0.113,"sire_track_interaction":"エスポワールシチー_良","bracket":2},
            {"horse_num":6, "horse_name":"ヴァルツァーシャル","sex_age":"牡7","weight_constraint":57.0,"jockey":"丹内祐次","trainer":"高木登",  "horse_weight":"514(0)", "prev_rank":1,"rest_days":42,"odds":4.9, "weather":"晴","track":"良","sire":"ゴールドアリュール","bms":"キングカメハメハ","lineage":"roberto",    "prev1_relative_last3f":0.8,"prev2_relative_last3f":0.7,"prev3_relative_last3f":0.6,"prev1_passage_num":2.0,"prev2_passage_num":2.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0,"weight_ratio":0.111,"sire_track_interaction":"ゴールドアリュール_良","bracket":3},
            {"horse_num":7, "horse_name":"アクションプラン",  "sex_age":"牡5","weight_constraint":57.0,"jockey":"荻野極",  "trainer":"相沢郁",  "horse_weight":"488(+6)","prev_rank":1,"rest_days":35,"odds":9.6, "weather":"晴","track":"良","sire":"ヘニーヒューズ",  "bms":"サウスヴィグラス","lineage":"storm_cat",  "prev1_relative_last3f":0.9,"prev2_relative_last3f":0.8,"prev3_relative_last3f":0.7,"prev1_passage_num":3.0,"prev2_passage_num":3.0,"prev3_passage_num":4.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5,"weight_ratio":0.117,"sire_track_interaction":"ヘニーヒューズ_良","bracket":3},
            {"horse_num":12,"horse_name":"ペイシャエス",      "sex_age":"牡8","weight_constraint":57.0,"jockey":"横山和生","trainer":"大久保龍志","horse_weight":"510(-4)","prev_rank":3,"rest_days":49,"odds":4.9, "weather":"晴","track":"良","sire":"エスポワールシチー","bms":"コマンダーインチーフ","lineage":"storm_cat","prev1_relative_last3f":0.5,"prev2_relative_last3f":0.4,"prev3_relative_last3f":0.6,"prev1_passage_num":4.0,"prev2_passage_num":4.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0,"weight_ratio":0.112,"sire_track_interaction":"エスポワールシチー_良","bracket":6},
            {"horse_num":15,"horse_name":"サンデーファンデー", "sex_age":"牡6","weight_constraint":56.0,"jockey":"角田大和","trainer":"音無秀孝","horse_weight":"468(-2)","prev_rank":4,"rest_days":56,"odds":12.8,"weather":"晴","track":"良","sire":"カジノドライヴ",  "bms":"サンデーサイレンス","lineage":"northern_dancer","prev1_relative_last3f":0.4,"prev2_relative_last3f":0.3,"prev3_relative_last3f":0.2,"prev1_passage_num":5.0,"prev2_passage_num":6.0,"prev3_passage_num":5.0,"is_long_trip":0,"is_stay":0,"weight_delta":-0.5,"weight_ratio":0.120,"sire_track_interaction":"カジノドライヴ_良","bracket":7},
        ],
    },

    "202607010611": {  # 高松宮記念 2026/3/29 中京 芝1200m
        "name": "高松宮記念(GI) 2026/3/29 中京 芝1200m",
        "surface": "turf", "distance": 1200, "venue": "中京",
        "model": "turf_short",
        "results": [
            {"rank":1, "horse_num":9, "horse_name":"サトノレーヴ",      "jockey":"C.ルメール","odds":3.5,  "popularity":1},
            {"rank":2, "horse_num":6, "horse_name":"レッドモンレーヴ",   "jockey":"酒井学",   "odds":67.0, "popularity":15},
            {"rank":3, "horse_num":8, "horse_name":"ウインカーネリアン", "jockey":"三浦皇成", "odds":19.4, "popularity":7},
            {"rank":4, "horse_num":1, "horse_name":"パンジャタワー",    "jockey":"松山弘平", "odds":5.0,  "popularity":3},
            {"rank":5, "horse_num":14,"horse_name":"レイピア",           "jockey":"丸山元気", "odds":19.8, "popularity":8},
        ],
        "entries": [
            {"horse_num":1, "horse_name":"パンジャタワー",   "sex_age":"牡4","weight_constraint":58.0,"jockey":"松山弘平", "trainer":"杉山晴紀","horse_weight":"478(0)", "prev_rank":1,"rest_days":42,"odds":5.0, "weather":"晴","track":"良","sire":"ロードカナロア","bms":"ハービンジャー",    "lineage":"storm_cat",       "prev1_relative_last3f":1.0,"prev2_relative_last3f":0.9,"prev3_relative_last3f":0.8,"prev1_passage_num":2.0,"prev2_passage_num":2.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5,"weight_ratio":0.121,"sire_track_interaction":"ロードカナロア_良","bracket":1},
            {"horse_num":6, "horse_name":"レッドモンレーヴ", "sex_age":"牡7","weight_constraint":58.0,"jockey":"酒井学",   "trainer":"石坂正",  "horse_weight":"486(+4)","prev_rank":5,"rest_days":56,"odds":67.0,"weather":"晴","track":"良","sire":"スクリーンヒーロー","bms":"ベーカバド",       "lineage":"roberto",         "prev1_relative_last3f":0.3,"prev2_relative_last3f":0.2,"prev3_relative_last3f":0.4,"prev1_passage_num":4.0,"prev2_passage_num":5.0,"prev3_passage_num":4.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0,"weight_ratio":0.119,"sire_track_interaction":"スクリーンヒーロー_良","bracket":3},
            {"horse_num":8, "horse_name":"ウインカーネリアン","sex_age":"牡7","weight_constraint":58.0,"jockey":"三浦皇成","trainer":"田島俊明","horse_weight":"490(-2)","prev_rank":3,"rest_days":35,"odds":19.4,"weather":"晴","track":"良","sire":"スクワートルスクワート","bms":"ゴールドヘイロー","lineage":"storm_cat",    "prev1_relative_last3f":0.7,"prev2_relative_last3f":0.6,"prev3_relative_last3f":0.5,"prev1_passage_num":3.0,"prev2_passage_num":3.0,"prev3_passage_num":4.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0,"weight_ratio":0.118,"sire_track_interaction":"スクワートル_良",    "bracket":4},
            {"horse_num":9, "horse_name":"サトノレーヴ",     "sex_age":"牡4","weight_constraint":58.0,"jockey":"C.ルメール","trainer":"堀宣行","horse_weight":"472(+2)","prev_rank":1,"rest_days":28,"odds":3.5, "weather":"晴","track":"良","sire":"ダイワメジャー","bms":"ディープインパクト",  "lineage":"northern_dancer", "prev1_relative_last3f":1.5,"prev2_relative_last3f":1.3,"prev3_relative_last3f":1.1,"prev1_passage_num":2.0,"prev2_passage_num":1.0,"prev3_passage_num":2.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5,"weight_ratio":0.123,"sire_track_interaction":"ダイワメジャー_良","bracket":5},
            {"horse_num":13,"horse_name":"ナムラクレア",     "sex_age":"牝6","weight_constraint":56.0,"jockey":"浜中俊",   "trainer":"松下武士","horse_weight":"450(0)", "prev_rank":2,"rest_days":42,"odds":4.0, "weather":"晴","track":"良","sire":"ミッキーアイル","bms":"クロフネ",          "lineage":"northern_dancer", "prev1_relative_last3f":0.8,"prev2_relative_last3f":0.9,"prev3_relative_last3f":0.7,"prev1_passage_num":2.0,"prev2_passage_num":2.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.0,"weight_ratio":0.124,"sire_track_interaction":"ミッキーアイル_良","bracket":7},
            {"horse_num":14,"horse_name":"レイピア",         "sex_age":"牡5","weight_constraint":58.0,"jockey":"丸山元気","trainer":"音無秀孝","horse_weight":"464(+6)","prev_rank":2,"rest_days":35,"odds":19.8,"weather":"晴","track":"良","sire":"アドマイヤムーン","bms":"キングスベスト",    "lineage":"northern_dancer", "prev1_relative_last3f":0.6,"prev2_relative_last3f":0.5,"prev3_relative_last3f":0.7,"prev1_passage_num":3.0,"prev2_passage_num":4.0,"prev3_passage_num":3.0,"is_long_trip":0,"is_stay":0,"weight_delta":0.5,"weight_ratio":0.125,"sire_track_interaction":"アドマイヤムーン_良","bracket":7},
        ],
    },
}


def predict_race(race_id: str, race_meta: dict) -> pd.DataFrame:
    mtype  = race_meta['model']
    model  = joblib.load(os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2.pkl'))
    meta   = joblib.load(os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl'))
    explainer = shap.TreeExplainer(model)
    feat_cols = meta['feature_cols']

    entries = race_meta['entries']
    df = pd.DataFrame(entries)

    # horse_weight_raw
    df['horse_weight_raw'] = df['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['bracket_num']      = df['bracket'].astype(int)
    df['odds_float']       = pd.to_numeric(df['odds'], errors='coerce')

    # JV統計（全データから）
    raw_pq = os.path.join(BASE_DIR, 'data', 'raw', 'scraped_results.parquet')
    raw = pd.read_parquet(raw_pq)

    if '着 順' in raw.columns:
        rk = raw['着 順']
        if isinstance(rk, pd.DataFrame): rk = rk.iloc[:, 0]
        raw['rank_num'] = pd.to_numeric(rk.astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    else:
        raw['rank_num'] = pd.to_numeric(raw.get('rank', pd.Series(0, index=raw.index)), errors='coerce')

    jcol = '騎手' if '騎手' in raw.columns else 'jockey'
    raw['venue_code'] = raw['race_id'].astype(str).str[4:6]
    raw['is_win']  = (raw['rank_num'] == 1).astype(float)
    raw['is_top2'] = (raw['rank_num'] <= 2).astype(float)
    raw['is_top3'] = (raw['rank_num'] <= 3).astype(float)
    global_win = raw['is_win'].mean()
    global_t3  = raw['is_top3'].mean()
    k = 10

    jv = raw.groupby([jcol, 'venue_code']).agg(
        jv_win_rate=('is_win','mean'), jv_top3_rate=('is_top3','mean'), jv_cnt=('is_win','count')
    ).reset_index().rename(columns={jcol:'jockey'})
    j  = raw.groupby(jcol).agg(j_win_rate=('is_win','mean'), j_top3_rate=('is_top3','mean')).reset_index().rename(columns={jcol:'jockey'})

    venue_code = str(race_id)[4:6]
    for i, row in df.iterrows():
        jk = row['jockey']
        jvr = jv[(jv['jockey']==jk)&(jv['venue_code']==venue_code)]
        jr  = j[j['jockey']==jk]
        if len(jvr):
            rc = jvr.iloc[0]['jv_cnt']
            df.at[i,'jv_win_rate']     = jvr.iloc[0]['jv_win_rate']
            df.at[i,'jv_top3_rate']    = jvr.iloc[0]['jv_top3_rate']
            df.at[i,'jv_win_rate_adj'] = (jvr.iloc[0]['jv_win_rate']*rc + global_win*k)/(rc+k)
        else:
            jw = jr.iloc[0]['j_win_rate'] if len(jr) else global_win
            df.at[i,'jv_win_rate']     = jw
            df.at[i,'jv_top3_rate']    = jr.iloc[0]['j_top3_rate'] if len(jr) else global_t3
            df.at[i,'jv_win_rate_adj'] = (global_win*k)/k
        df.at[i,'j_win_rate']  = jr.iloc[0]['j_win_rate']  if len(jr) else global_win
        df.at[i,'j_top3_rate'] = jr.iloc[0]['j_top3_rate'] if len(jr) else global_t3
        df.at[i,'v_win_rate']  = global_win

        hw = float(str(row['horse_weight']).split('(')[0].strip())
        df.at[i,'weight_burden_score'] = row['weight_constraint'] / (hw+1e-5)
        df.at[i,'is_inner_bracket']     = 1 if int(row['bracket'])<=2 else 0
        df.at[i,'is_outer_bracket']     = 1 if int(row['bracket'])>=7 else 0
        is_local = 1 if int(venue_code)>=30 else 0
        df.at[i,'dirt_bracket_score']   = (3-row['bracket'])*0.02 if is_local else (row['bracket']-4.5)*0.015
        df.at[i,'bv_win_rate_adj']      = global_win

    # カテゴリエンコード
    le_path = os.path.join(BASE_DIR, 'src', 'features', 'encoders_v3.pkl')
    if os.path.exists(le_path):
        les = joblib.load(le_path)
        for col, le in les.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['unknown'])[0]
                    if 'unknown' in le.classes_ else 0
                )
    else:
        for col in ['sex_age','jockey','trainer','weather','track','sire','bms','lineage','sire_track_interaction']:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes

    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    df['pred_prob'] = model.predict_proba(X)[:, 1]
    df['ev']        = df['pred_prob'] * df['odds_float']

    # SHAP
    sv = explainer.shap_values(X)
    if isinstance(sv, list): sv = sv[1]
    df['_shap'] = [sorted(zip(feat_cols,s), key=lambda x: abs(x[1]), reverse=True)[:5] for s in sv]

    return df.sort_values('ev', ascending=False).reset_index(drop=True)


def backtest_report(race_id: str, race_meta: dict) -> str:
    df      = predict_race(race_id, race_meta)
    results = {r['horse_name']: r for r in race_meta['results']}
    lines   = []

    lines.append(f"\n{'═'*68}")
    lines.append(f"  📊 バックテスト: {race_meta['name']}")
    lines.append(f"  使用モデル: {race_meta['model']}")
    lines.append(f"{'═'*68}")
    lines.append(f"  {'馬番':>3} {'馬名':<16} {'実際着順':>6} {'EV':>6} {'確率':>6} {'オッズ':>6}  AI評価")
    lines.append(f"  {'─'*66}")

    total_bet = 0; total_return = 0
    for _, row in df.iterrows():
        name = row['horse_name']
        ev   = row['ev']
        prob = row['pred_prob']*100
        odds = row['odds_float']
        res  = results.get(name, {})
        actual_rank = res.get('rank', '?')

        mark = ''
        if   ev >= 2.0: mark = '★★★ 本命'
        elif ev >= 1.5: mark = '★★  要注目'
        elif ev >= 1.0: mark = '★   押さえ'

        rank_str = f"{actual_rank}着" if actual_rank != '?' else '圏外'
        lines.append(f"  {int(row['horse_num']):>3} {name:<16} {rank_str:>6} {ev:>6.2f} {prob:>5.1f}% {odds:>6.1f}倍  {mark}")

        if ev >= 1.5:
            total_bet += 100
            if actual_rank != '?' and actual_rank == 1:
                total_return += int(odds * 100)

    lines.append(f"\n  ◆ 確定着順TOP5:")
    for r in sorted(race_meta['results'], key=lambda x: x['rank'])[:5]:
        pred_row = df[df['horse_name']==r['horse_name']]
        ev_str   = f"EV={pred_row.iloc[0]['ev']:.2f}" if len(pred_row) else "EV=N/A"
        lines.append(f"    {r['rank']}着 {r['horse_name']:<14} オッズ{r['odds']}倍 人気{r['popularity']}番  AI予測: {ev_str}")

    if total_bet > 0:
        roi = total_return / total_bet * 100
        lines.append(f"\n  💰 EV≥1.5フィルター収支: 投資¥{total_bet} → 回収¥{total_return} (回収率 {roi:.1f}%)")
    lines.append(f"{'═'*68}")
    return '\n'.join(lines)


def main():
    print("=" * 68)
    print("  🏇 先週 3/28・29 バックテスト（三冠馬エンジン 答え合わせ）")
    print("=" * 68)
    for rid, meta in BACKTEST_RACES.items():
        print(backtest_report(rid, meta))


if __name__ == '__main__':
    main()
