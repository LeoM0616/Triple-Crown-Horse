import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import io
import re

# 保存先パス
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'data', 'raw')
output_path = os.path.join(data_dir, 'scraped_results.csv')

def scrape_race_results(race_id_list):
    """
    複数レースIDからネット競馬のレース結果をスクレイピングする
    """
    all_results = []
    
    for idx, race_id in enumerate(race_id_list):
        if idx % 10 == 0:
            print(f"Scraping progress: {idx}/{len(race_id_list)}")
            
        url = f"https://db.netkeiba.com/race/{race_id}/"
        
        try:
            # サーバー負荷軽減のため必ずスリープ (0.5秒)
            time.sleep(0.5)
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers)
            # 文字化け対策（netkeibaはEUC-JPが使われていることが多い）
            response.encoding = 'euc-jp'
            html = response.text
            
            # BeautifulSoupで基本情報の取得
            soup = BeautifulSoup(html, 'html.parser')
            info_p = soup.find('p', class_='smalltxt')
            race_info = info_p.text if info_p else ""
            
            # 日付の抽出
            date_match = re.search(r'(\d+)年(\d+)月(\d+)日', race_info)
            if date_match:
                race_date = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
            else:
                race_date = ""

            # 天候と馬場状態の抽出
            intro = soup.find('div', class_='data_intro')
            weather = 'unknown'
            track = 'unknown'
            if intro:
                intro_text = intro.text
                w_match = re.search(r'天候 : ([^\s/]+)', intro_text)
                t_match = re.search(r'(芝|ダ)[^\s]* : ([^\s/]+)', intro_text)
                if w_match: weather = w_match.group(1)
                if t_match: track = t_match.group(2)
            
            # pandasのread_htmlでメインの着順テーブルを取得
            dfs = pd.read_html(io.StringIO(html))
            
            if len(dfs) > 0:
                df = dfs[0]
                df['race_id'] = race_id
                df['race_info'] = race_info
                df['race_date'] = race_date
                df['weather'] = weather
                df['track'] = track
                
                # horse_id, 通過, 上り の抽出
                table = soup.find('table', class_='race_table_01')
                if table:
                    # horse_id
                    horse_links = table.find_all('a', href=re.compile(r'/horse/\d+'))
                    horse_ids = [re.search(r'/horse/(\d+)', a['href']).group(1) for a in horse_links]
                    if len(horse_ids) == len(df):
                        df['horse_id'] = horse_ids
                    else:
                        df['horse_id'] = ""
                        
                    # passage, last3f
                    trs = table.find_all('tr')
                    passage_list = []
                    last3f_list = []
                    if len(trs) > 0:
                        headers_row = [th.text.strip() for th in trs[0].find_all(['th', 'td'])]
                        passage_idx = -1
                        last3f_idx = -1
                        for i, h in enumerate(headers_row):
                            if '通過' in h:
                                passage_idx = i
                            elif '上り' in h:
                                last3f_idx = i
                                
                        for tr in trs[1:]:
                            tds = tr.find_all('td')
                            if len(tds) > 5: # データ行のみ
                                p_val = tds[passage_idx].text.strip() if passage_idx >= 0 and len(tds) > passage_idx else ""
                                l_val = tds[last3f_idx].text.strip() if last3f_idx >= 0 and len(tds) > last3f_idx else ""
                                passage_list.append(p_val)
                                last3f_list.append(l_val)
                                
                    if len(passage_list) == len(df):
                        df['passage_rank'] = passage_list
                    else:
                        df['passage_rank'] = ""
                        
                    if len(last3f_list) == len(df):
                        df['last3f_time'] = last3f_list
                    else:
                        df['last3f_time'] = ""
                        
                else:
                    df['horse_id'] = ""
                    df['passage_rank'] = ""
                    df['last3f_time'] = ""
                    
                all_results.append(df)
            else:
                print(f"Warning: テーブルが見つかりませんでした (race_id: {race_id})")
        
        except Exception as e:
            print(f"Error scraping {race_id}: {e}")
            
    if all_results:
        # すべてのレース結果を結合
        final_df = pd.concat(all_results, ignore_index=True)
        # フォルダが存在しない場合は作成
        os.makedirs(data_dir, exist_ok=True)
        # CSVとして保存
        final_df.to_csv(output_path, index=False)
        print(f"\nScraping complete. Saved to: {output_path}")
        print(f"Total rows: {len(final_df)}")
        return final_df
    else:
        print("No data collected.")
        return None

if __name__ == '__main__':
    from get_graded_races import fetch_graded_race_ids
    
    print("Fetching graded race IDs for 2023-2026...")
    race_ids = fetch_graded_race_ids(2023, 2026)
    
    if race_ids:
        print(f"Start scraping {len(race_ids)} races (This will take a few minutes)...")
        df = scrape_race_results(race_ids)
        if df is not None:
            print("\n=== サンプル確認 ===")
            try:
                print(df.head())
            except UnicodeEncodeError:
                pass # コンソールの文字化け回避
    else:
        print("No race IDs found to scrape.")
