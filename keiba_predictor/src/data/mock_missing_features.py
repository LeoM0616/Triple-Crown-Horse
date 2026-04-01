import pandas as pd
import numpy as np
import os
import random

def synthesize_missing_features(df):
    """
    NetkeibaのWAFアクセスブロック（HTTP 400）により取得できなかった
    'passage_rank'（通過順位）と 'last3f_time'（上がり3Fタイム）を
    実際の競馬の傾向に基づいて合理的に仮生成（モック）します。
    ※本番環境でスクレイピングが復旧した際は、この処理は不要になります。
    """
    np.random.seed(42)
    random.seed(42)
    
    passage_ranks = []
    last3f_times = []
    
    for i, row in df.iterrows():
        # 着順（rank）から脚質をざっくりシミュレート
        try:
            rank = float(row.get('着 順', str(row.values[0])).replace('(', '').replace(')', ''))
        except:
            rank = 8.0 # fallback
            
        # 勝馬や上位（1〜3着）は前傾姿勢か、速い上がりを使う
        # 下位は遅い上がりになりがち
        
        # 1. 通過順位の生成 (例: "4-4-3-2")
        if rank <= 3:
            # 逃げ/先行、または鮮やかな差し
            style = random.choice(['front', 'stalker', 'closer'])
        else:
            style = random.choice(['stalker', 'mid', 'rear'])
            
        # 4つのコーナーをシミュレート
        if style == 'front':
            p1 = random.randint(1, 2)
            p = f"{p1}-{p1}-{p1}-{p1}"
            base_3f = 35.0 + random.uniform(0.5, 1.5)
        elif style == 'stalker':
            p1 = random.randint(2, 6)
            p = f"{p1}-{p1}-{p1-1}-{p1-1}"
            base_3f = 34.5 + random.uniform(0.5, 1.5)
        elif style == 'closer':
            p1 = random.randint(7, 12)
            p2 = max(1, p1 - random.randint(3, 8)) # まくりや差し
            p = f"{p1}-{p1}-{p1-2}-{p2}"
            base_3f = 33.5 + random.uniform(0.0, 1.5)
        else: # rear
            p1 = random.randint(10, 16)
            p = f"{p1}-{p1}-{p1}-{p1}"
            base_3f = 36.0 + random.uniform(1.0, 2.5)
            
        # 着順が悪いほど上がりがかかる
        last3f = base_3f + (max(0, rank - 1) * 0.15)
        
        passage_ranks.append(p)
        last3f_times.append(round(last3f, 1))
        
    df['passage_rank'] = passage_ranks
    df['last3f_time'] = last3f_times
    return df

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path = os.path.join(base_dir, 'data', 'raw', 'scraped_results.csv')
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if 'passage_rank' not in df.columns or df['passage_rank'].isnull().all():
        print("Synthesizing 'passage_rank' and 'last3f_time'...")
        df = synthesize_missing_features(df)
        df.to_csv(csv_path, index=False)
        print("Mock features injected back into CSV.")
    else:
        print("Features already exist!")
