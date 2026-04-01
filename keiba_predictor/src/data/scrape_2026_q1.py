"""
2026年Q1データ（1-3月）の追加スクレイピングと全データ統合・前処理スクリプト
"""
import sys, os, time, re, io
import pandas as pd
import requests
from bs4 import BeautifulSoup

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_PQ    = os.path.join(BASE_DIR, 'data', 'raw', 'scraped_results.parquet')
RAW_CSV   = os.path.join(BASE_DIR, 'data', 'raw', 'scraped_results.csv')
HEADERS   = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def fetch_2026_race_ids() -> list:
    """2026年1〜3月のレースIDを取得"""
    race_ids = []
    for page in range(1, 30):
        url = (
            "https://db.netkeiba.com/?pid=race_list"
            "&start_year=2026&start_mon=1&end_year=2026&end_mon=3"
            "&list=100"
            "&grade%5b%5d=1&grade%5b%5d=2&grade%5b%5d=3"
            "&grade%5b%5d=4&grade%5b%5d=5&grade%5b%5d=10"
            f"&page={page}"
        )
        time.sleep(1.2)
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.encoding = 'euc-jp'
            soup = BeautifulSoup(r.text, 'html.parser')
            links = soup.find_all('a', href=re.compile(r'/race/\d{12}'))
            ids = list({re.search(r'/race/(\d{12})', a['href']).group(1) for a in links})
            if not ids:
                break
            race_ids.extend(ids)
            print(f"  Page {page}: {len(ids)} races found")
        except Exception as e:
            print(f"  Page {page}: Error - {e}")
            break
    return list(set(race_ids))


def scrape_race(race_id: str) -> list:
    """1レースをスクレイプして行リストを返す"""
    url = f"https://db.netkeiba.com/race/{race_id}/"
    time.sleep(0.6)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        info_p   = soup.find('p', class_='smalltxt')
        race_info = info_p.text.strip() if info_p else ""
        dm = re.search(r'(\d+)年(\d+)月(\d+)日', race_info)
        race_date = f"{dm.group(1)}-{dm.group(2).zfill(2)}-{dm.group(3).zfill(2)}" if dm else ""

        intro = soup.find('div', class_='data_intro')
        weather, track = 'unknown', 'unknown'
        if intro:
            t = intro.text
            wm = re.search(r'天候 : ([^\s/]+)', t)
            tm = re.search(r'(芝|ダ)[^\s]* : ([^\s/]+)', t)
            if wm: weather = wm.group(1)
            if tm: track   = tm.group(2)

        dfs = pd.read_html(io.StringIO(r.text))
        result_df = None
        for df in dfs:
            if '着順' in df.columns or '着 順' in df.columns:
                result_df = df
                break
        if result_df is None:
            return []

        # 馬IDと上がり・通過順位の取得
        horse_ids, passage_ranks, last3fs = [], [], []
        links = soup.select('table.race_table_01 a[href*="/horse/"]')
        for a in links:
            m = re.search(r'/horse/(\d+)/', a['href'])
            if m:
                horse_ids.append(m.group(1))

        rows_out = []
        for i, row in result_df.iterrows():
            d = row.to_dict()
            d['race_id']   = race_id
            d['race_info'] = race_info
            d['race_date'] = race_date
            d['weather']   = weather
            d['track']     = track
            d['horse_id']  = horse_ids[i] if i < len(horse_ids) else ''
            # passage_rank / last3f_time は列名で取得
            d['passage_rank'] = row.get('通過', row.get('通 過', ''))
            d['last3f_time']  = row.get('上り', row.get('上 り', ''))
            rows_out.append(d)
        return rows_out

    except Exception as e:
        return []


def main():
    # 既存データ読み込み
    existing = pd.read_parquet(RAW_PQ)
    existing_ids = set(existing['race_id'].astype(str).unique())
    existing_2026 = existing[existing['race_id'].astype(str).str[:4] == '2026']
    print(f"既存データ: {len(existing)} rows / 2026年: {existing_2026['race_id'].nunique()} レース")

    # 2026年1-3月のレースID取得
    print("\n2026年Q1 レースID取得中...")
    new_ids = fetch_2026_race_ids()
    to_scrape = [rid for rid in new_ids if rid not in existing_ids]
    print(f"  総取得: {len(new_ids)} / 未取得(追加対象): {len(to_scrape)}")

    if not to_scrape:
        print("  既に全件取得済みです。前処理へ進みます。")
    else:
        # スクレイプ
        new_rows = []
        for i, rid in enumerate(to_scrape):
            rows = scrape_race(rid)
            new_rows.extend(rows)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(to_scrape)}] scraped: {len(new_rows)} rows")

        print(f"\n  新規取得: {len(new_rows)} rows")
        if new_rows:
            new_df   = pd.DataFrame(new_rows)
            # race_id を文字列に統一
            existing['race_id'] = existing['race_id'].astype(str)
            new_df['race_id']   = new_df['race_id'].astype(str)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['race_id', '馬 番'], inplace=True)
            new_df['race_id']   = new_df['race_id'].astype(str)
            existing['race_id'] = existing['race_id'].astype(str)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['race_id', '馬 番'], inplace=True)

            # CSVで保存し、既存のparquetはCSVから再生成
            combined.to_csv(RAW_CSV, index=False, encoding='utf-8-sig')

            # parquet保存: 型を全て文字列か数値に統一
            pq_df = combined.copy()
            for col in pq_df.columns:
                try:
                    pq_df[col] = pd.to_numeric(pq_df[col], errors='ignore')
                except Exception:
                    pass
                if pq_df[col].dtype == object:
                    pq_df[col] = pq_df[col].astype(str).replace('nan', '')
            pq_df.to_parquet(RAW_PQ, index=False)
            print(f"  保存完了: {len(combined)} rows total")

    print("\nScraping完了。次のステップ: preprocess & retrain")


if __name__ == '__main__':
    main()
