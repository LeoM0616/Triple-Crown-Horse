import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
import re

def fetch_horse_pedigree():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'raw', 'scraped_results.csv')
    output_path = os.path.join(base_dir, 'data', 'raw', 'horse_pedigree.csv')

    if not os.path.exists(input_path):
        print("scraped_results.csv not found.")
        return
        
    df = pd.read_csv(input_path)
    if 'horse_id' not in df.columns:
        print("horse_id column missing. Run scrape_race_results.py first.")
        return
        
    horse_ids = df['horse_id'].dropna().astype(str).unique()
    horse_ids = [hid for hid in horse_ids if hid and hid.lower() != 'nan' and hid.isdigit()]
    
    # 既にスクレイピングしたデータがあればスキップする構造に（途中再開用）
    existing_ids = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, dtype={'horse_id': str})
        existing_ids = set(existing_df['horse_id'].dropna().astype(str))
        all_pedigrees = existing_df.to_dict('records')
    else:
        all_pedigrees = []
        
    target_ids = [hid for hid in horse_ids if hid not in existing_ids]
    print(f"Start scraping pedigree for {len(target_ids)} unique horses... (Total: {len(horse_ids)})")
    
    for idx, hid in enumerate(target_ids):
        if idx % 50 == 0 and idx > 0:
            print(f"Pedigree progress: {idx}/{len(target_ids)}")
            # 途中保存
            pd.DataFrame(all_pedigrees).to_csv(output_path, index=False)
            
        url = f"https://db.netkeiba.com/horse/ped/{hid}/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        sire = 'unknown'
        bms = 'unknown'
        lineage = 'unknown'
        
        try:
            time.sleep(0.1) # 高速＆安全な0.1秒スリープ
            response = requests.get(url, headers=headers)
            response.encoding = 'euc-jp'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            b = soup.find('table', class_='blood_table')
            if b:
                trs = b.find_all('tr')
                if len(trs) >= 17:
                    # 父
                    sire_td = trs[0].find_all('td')[0]
                    sire = sire_td.text.strip().split('\n')[0].strip()
                    lin_m = re.search(r'([A-Za-z0-9_]+)系', sire_td.text)
                    if lin_m:
                        lineage = lin_m.group(1) + '系'
                    
                    # 母父
                    bms_tds = trs[16].find_all('td')
                    if len(bms_tds) > 1:
                        bms = bms_tds[1].text.strip().split('\n')[0].strip()
                        
            all_pedigrees.append({
                'horse_id': hid,
                'sire': sire,
                'bms': bms,
                'lineage': lineage
            })
        except Exception as e:
            all_pedigrees.append({
                'horse_id': hid,
                'sire': 'error',
                'bms': 'error',
                'lineage': 'error'
            })
            
    if all_pedigrees:
        ped_df = pd.DataFrame(all_pedigrees)
        ped_df.to_csv(output_path, index=False)
        print(f"Saved pedigree for {len(ped_df)} horses to {output_path}")
    else:
        print("No pedigrees collected.")

if __name__ == '__main__':
    fetch_horse_pedigree()
