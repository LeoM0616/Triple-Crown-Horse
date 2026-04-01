import requests
import pandas as pd
import io
import time
import os

def fetch_horse_history():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'raw', 'scraped_results.csv')
    output_path = os.path.join(base_dir, 'data', 'raw', 'horse_histories.csv')

    if not os.path.exists(input_path):
        print("scraped_results.csv not found.")
        return
        
    df = pd.read_csv(input_path)
    if 'horse_id' not in df.columns:
        print("horse_id column missing. Run scrape_race_results.py first.")
        return
        
    horse_ids = df['horse_id'].dropna().astype(str).unique()
    horse_ids = [hid for hid in horse_ids if hid and hid.lower() != 'nan' and hid.isdigit()]
    
    print(f"Start scraping histories for {len(horse_ids)} unique horses...")
    
    all_histories = []
    
    for idx, hid in enumerate(horse_ids):
        if idx % 50 == 0:
            print(f"Horse progress: {idx}/{len(horse_ids)}")
            
        url = f"https://db.netkeiba.com/horse/{hid}/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            # 高速化（0.1秒スリープ）
            time.sleep(0.1)
            response = requests.get(url, headers=headers)
            response.encoding = 'euc-jp'
            
            # 過去レースのあるテーブルを抽出
            dfs = pd.read_html(io.StringIO(response.text), match='日付', flavor='lxml')
            if dfs:
                history_df = dfs[0]
                history_df['horse_id'] = hid
                all_histories.append(history_df)
        except Exception as e:
            # lxmlエラー対策としてhtml5lib等がない場合はpassして無視
            pass
            
    if all_histories:
        final_history = pd.concat(all_histories, ignore_index=True)
        final_history.to_csv(output_path, index=False)
        print(f"Saved {len(final_history)} past races to {output_path}")
    else:
        print("No histories collected.")

if __name__ == '__main__':
    fetch_horse_history()
