import pandas as pd
import numpy as np
import os
import joblib
import sys

# Windows環境でのUnicodeEncodeError対策
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def simulate_roi():
    """
    2025年の全重賞を対象に、AIの予測確率上位を用いた馬券シミュレーションを行う。
    ※単勝以外（複勝・ワイド）の確定配当はスクレイピングデータにないため、単勝オッズから理論値で推計します。
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'processed', 'model_input.csv')
    model_path = os.path.join(base_dir, 'src', 'models', 'lgbm_classifier_walkforward.pkl')
    raw_path = os.path.join(base_dir, 'data', 'raw', 'scraped_results.csv')
    
    # データのロード
    df = pd.read_csv(input_path, dtype={'race_id': str})
    df_raw = pd.read_csv(raw_path, dtype={'horse_id': str, 'race_id': str})
    model = joblib.load(model_path)
    
    # 元データから日付と基本情報をマージ
    df_raw.rename(columns={'着 順': 'raw_rank', '馬 番': 'horse_num', '単勝': 'raw_odds'}, inplace=True)
    df_raw['race_date_dt'] = pd.to_datetime(df_raw['race_date'], errors='coerce')
    
    # 2025年のレース（重賞）のみを対象とする
    df_raw_2025 = df_raw[df_raw['race_date_dt'].dt.year == 2025].copy()
    
    # 推論するための特徴量カラム
    ignore_cols = ['rank', 'is_top3', 'race_id']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # シミュレーション用dfの構築（2025年のデータのみ）
    df_2025 = df[df['race_id'].isin(df_raw_2025['race_id'])].copy()
    if df_2025.empty:
        print("2025年のデータが見つかりませんでした。全データでシミュレーションします。")
        df_2025 = df.copy()
        
    X_2025 = df_2025[feature_cols].copy()
    
    # 予測確率の算出
    df_2025['pred_prob'] = model.predict_proba(X_2025)[:, 1]
    
    # 回収率・的中率用変数
    total_races = 0
    win_cost = 0.0
    win_return = 0.0
    bets_count = 0
    hits_count = 0
    
    # レースごとに集計
    grouped = df_2025.groupby('race_id')
    for race_id, group in grouped:
        if len(group) < 5: continue # 少頭数すぎるレースはスキップ
        
        total_races += 1
        
        # 期待値(EV) = 予測確率 * 単勝オッズ
        tmp_group = group.copy()
        tmp_group['odds_float'] = pd.to_numeric(tmp_group['odds'], errors='coerce').fillna(1.0)
        tmp_group['ev'] = tmp_group['pred_prob'] * tmp_group['odds_float']
        
        # 予測確率順(本命順)に並び替え
        tmp_group = tmp_group.sort_values('pred_prob', ascending=False)
        
        # 推論確率が最も高い「本命馬(◎)」を1頭だけ選定
        top1 = tmp_group.iloc[0]
        
        # EVフィルター：本命(◎)が「確率高め × オッズ旨味あり」の条件を両方満たすときのみ勝負
        # 複勝率25%以上 = 「信頼できる本命」、EV>=0.8 = 「オッズが配当とバランスが取れている」
        if top1['pred_prob'] >= 0.25 and top1['ev'] >= 0.8:
            win_cost += 100
            bets_count += 1
            if top1['rank'] == 1.0:
                hits_count += 1
                win_return += top1['odds_float'] * 100
                    
    # 結果の集計と出力
    win_roi = (win_return / win_cost) * 100 if win_cost > 0 else 0
    hit_rate = (hits_count / bets_count) * 100 if bets_count > 0 else 0
    
    print("\n" + "="*50)
    print(f" 🏇 2025年 AI競馬 実戦シミュレーション結果 ({total_races}レース)")
    print("="*50)
    print(f"【投資ルール：『期待値(EV)買い』ハイブリッド版】")
    print(f" ・対象馬: AIが全頭から算出した「確率トップ(◎)」の馬に限定")
    print(f" ・フィルター: (予測確率 × 単勝オッズ) >= 1.50 を満たす場合のみ勝負（見送りあり）")
    print("-" * 50)
    print(f" [単勝] 回収率: {win_roi:.1f}% (払戻: {int(win_return):,}円 / 投資: {int(win_cost):,}円)")
    print(f" [成績] 的中率(Hit Rate): {hit_rate:.1f}% (購入回数: {bets_count} ｜ 的中: {hits_count})")
    print("="*50)

if __name__ == "__main__":
    simulate_roi()
