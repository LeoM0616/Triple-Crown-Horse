import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, os

BASE = '.'
csv = os.path.join(BASE, 'data', 'raw', 'scraped_results.csv')
if os.path.exists(csv):
    df = pd.read_csv(csv)
    print(f"CSV rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()[:15]}")
    print(f"\nRace entry counts:")
    ec = df.groupby('race_id')['race_id'].count()
    print(ec.describe())
    print(ec.value_counts().sort_index().head(10))
    print(f"\n年別:")
    df['year'] = df['race_id'].astype(str).str[:4]
    print(df['year'].value_counts().sort_index())
    print(f"\n'rank' or '着 順' sample:")
    rank_col = '着 順' if '着 順' in df.columns else 'rank'
    print(rank_col, df[rank_col].value_counts().sort_index().head(10))
else:
    print("CSV not found!")
