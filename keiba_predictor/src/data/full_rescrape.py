"""
全件再スクレイプ + 増分保存スクリプト（2023-2026/03）
- 既存データとマージせず、クリーンなparquetとして再生成
- 500件ごとに中間保存（クラッシュ対策）
- 完了後に preprocess → retrain まで自動実行
"""
import sys, os, time, re, io
import pandas as pd
import requests
from bs4 import BeautifulSoup

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_CSV  = os.path.join(DATA_DIR, 'scraped_results_full.csv')
OUT_PQ   = os.path.join(DATA_DIR, 'scraped_results_full.parquet')
CHKPT    = os.path.join(DATA_DIR, 'scrape_checkpoint.txt')
HEADERS  = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def fetch_race_ids_2023_2026() -> list:
    from src.data.get_graded_races import fetch_graded_race_ids
    return fetch_graded_race_ids(2023, 2026)


def scrape_one(race_id: str) -> list:
    url = f"https://db.netkeiba.com/race/{race_id}/"
    time.sleep(0.55)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = 'euc-jp'
        soup = BeautifulSoup(r.text, 'html.parser')

        info_p = soup.find('p', class_='smalltxt')
        race_info = info_p.text.strip() if info_p else ""
        dm = re.search(r'(\d+)年(\d+)月(\d+)日', race_info)
        race_date = f"{dm.group(1)}-{dm.group(2).zfill(2)}-{dm.group(3).zfill(2)}" if dm else ""

        intro = soup.find('div', class_='data_intro')
        weather, track = '晴', '良'
        if intro:
            t = intro.text
            wm = re.search(r'天候 : ([^\s/]+)', t)
            tm = re.search(r'(芝|ダ)[^\s]* : ([^\s/]+)', t)
            if wm: weather = wm.group(1)
            if tm: track   = tm.group(2)

        dfs = pd.read_html(io.StringIO(r.text))
        df = None
        for d in dfs:
            cols = [str(c) for c in d.columns]
            if any('着' in c for c in cols) or '着順' in d.columns or '着 順' in d.columns:
                df = d; break
        if df is None and dfs:
            df = dfs[0]
        if df is None:
            return []

        # 追加列
        df['race_id']   = str(race_id)
        df['race_info'] = race_info
        df['race_date'] = race_date
        df['weather']   = weather
        df['track']     = track

        # horse_id, passage, last3f
        table = soup.find('table', class_='race_table_01')
        df['horse_id'] = ""
        df['passage_rank'] = ""
        df['last3f_time']  = ""
        if table:
            hl = table.find_all('a', href=re.compile(r'/horse/\d+'))
            hids = [re.search(r'/horse/(\d+)', a['href']).group(1) for a in hl]
            if len(hids) == len(df):
                df['horse_id'] = hids

            trs = table.find_all('tr')
            if trs:
                hdrs = [th.text.strip() for th in trs[0].find_all(['th','td'])]
                pidx = next((i for i,h in enumerate(hdrs) if '通過' in h), -1)
                lidx = next((i for i,h in enumerate(hdrs) if '上り' in h), -1)
                pl, ll = [], []
                for tr in trs[1:]:
                    tds = tr.find_all('td')
                    if len(tds) > 5:
                        pl.append(tds[pidx].text.strip() if pidx>=0 and len(tds)>pidx else "")
                        ll.append(tds[lidx].text.strip() if lidx>=0 and len(tds)>lidx else "")
                if len(pl) == len(df):
                    df['passage_rank'] = pl
                    df['last3f_time']  = ll

        return df.to_dict('records')
    except Exception as e:
        return []


def load_checkpoint() -> set:
    if os.path.exists(CHKPT):
        with open(CHKPT) as f:
            return set(f.read().splitlines())
    return set()


def save_checkpoint(done_ids: set):
    with open(CHKPT, 'w') as f:
        f.write('\n'.join(sorted(done_ids)))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 既存の完全データがあれば読む
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV, dtype=str)
        done_ids = set(existing['race_id'].astype(str).unique())
        all_rows = existing.to_dict('records')
        print(f"既存: {len(existing)} rows, {len(done_ids)} races 読込済み")
    else:
        done_ids = load_checkpoint()
        all_rows = []
        print(f"チェックポイント: {len(done_ids)} races 処理済み")

    # race_idリスト取得
    print("レースIDリスト取得中 (2023-2026/03)...")
    sys.path.insert(0, BASE_DIR)
    from src.data.get_graded_races import fetch_graded_race_ids
    all_ids = fetch_graded_race_ids(2023, 2026)
    todo = [rid for rid in all_ids if str(rid) not in done_ids]
    print(f"総レース: {len(all_ids)} | 未取得: {len(todo)}")

    for i, rid in enumerate(todo):
        rows = scrape_one(rid)
        if rows:
            all_rows.extend(rows)
            done_ids.add(str(rid))

        # 100件ごとに中間保存
        if (i + 1) % 100 == 0:
            df_tmp = pd.DataFrame(all_rows)
            df_tmp.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
            save_checkpoint(done_ids)
            print(f"  [{i+1}/{len(todo)}] {len(all_rows)} rows 保存済み")

    # 最終保存
    if all_rows:
        df_final = pd.DataFrame(all_rows)
        df_final.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        save_checkpoint(done_ids)
        print(f"\n✅ 完了: {len(df_final)} rows, {df_final['race_id'].nunique()} races")
        print(f"保存先: {OUT_CSV}")

        # parquetも生成
        for col in df_final.columns:
            if df_final[col].dtype.name == 'int64':
                df_final[col] = df_final[col].astype(str)
        df_final.to_parquet(OUT_PQ, index=False)
        print(f"Parquet: {OUT_PQ}")

        # 元のparquetも置き換え
        import shutil
        dst_pq  = os.path.join(DATA_DIR, 'scraped_results.parquet')
        dst_csv = os.path.join(DATA_DIR, 'scraped_results.csv')
        shutil.copy2(OUT_PQ,  dst_pq)
        shutil.copy2(OUT_CSV, dst_csv)
        print("✅ scraped_results.parquet/.csv を完全版で置き換えました")

    print("\n次: python src/features/preprocess.py && python src/models/rebuild_final.py")


if __name__ == '__main__':
    main()
