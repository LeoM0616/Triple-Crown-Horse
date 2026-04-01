import requests
from bs4 import BeautifulSoup
import re
import time

def fetch_graded_race_ids(start_year, end_year):
    """
    指定年（例:2023〜2026）の対象レース（G1〜3、OP、L、3勝クラス）のレースIDを取得。
    """
    race_ids = []
    
    # データ量を十分に確保するため、上限100ページまでループして検索
    for page in range(1, 100):
        # grade=1,2,3 (G1, G2, G3) + 4 (OP), 5 (3勝クラス), 10 (L)
        # list=100 (1ページ100件表示)
        url = f"https://db.netkeiba.com/?pid=race_list&start_year={start_year}&start_mon=1&end_year={end_year}&end_mon=12&list=100&grade%5b%5d=1&grade%5b%5d=2&grade%5b%5d=3&grade%5b%5d=4&grade%5b%5d=5&grade%5b%5d=10&page={page}"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        time.sleep(1)  # サーバー負荷軽減のため1秒スリープ
        print(f"Fetching race IDs from search page {page}...")
        
        try:
            response = requests.get(url, headers=headers)
            response.encoding = 'euc-jp'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # /race/12桁の数字/ へのリンクを探す
            links = soup.find_all('a', href=re.compile(r'/race/\d{12}'))
            page_ids = []
            for a in links:
                match = re.search(r'/race/(\d{12})', a['href'])
                if match:
                    page_ids.append(match.group(1))
                    
            page_ids = list(set(page_ids)) # 重複除去
            
            if not page_ids:
                # このページの結果にレースが無ければ最後尾と判断してループ終了
                break
                
            race_ids.extend(page_ids)
            print(f"-> Found {len(page_ids)} race IDs.")
            
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
            
    # 全体での重複除去
    race_ids = list(set(race_ids))
    print(f"\nTotal unique target race IDs found: {len(race_ids)}")
    return race_ids

if __name__ == "__main__":
    ids = fetch_graded_race_ids(2023, 2026)
    print(f"Sample: {ids[:5]}")
