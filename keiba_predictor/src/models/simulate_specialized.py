"""
Specialized model simulation:
各レースを model_type (turf_short/turf_long/dirt) に振り分け、
専用モデルで予測確率を計算。EV >= 1.5 フィルターで単勝を購入。
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# train_specialized.py と同じ分類ロジックをインポート
sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type

def simulate_specialized(ev_thresh: float = 1.5):
    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    proc_pq    = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')
    raw_pq     = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    model_dir  = os.path.join(base_dir, 'src', 'models')

    df  = pd.read_parquet(proc_pq)
    raw = pd.read_parquet(raw_pq)[['race_id','race_info']].drop_duplicates('race_id')
    df['race_id'] = df['race_id'].astype(str)
    raw['race_id'] = raw['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    # モデルタイプを付与
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['year']       = df['race_id'].str[:4].astype(int)

    # 2025年のみテスト対象
    df_2025 = df[df['year'] >= 2025].copy()

    ignore_cols  = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance', 'model_type', 'race_info']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # モデルをロード
    models = {}
    for mtype in ['turf_short', 'turf_long', 'dirt']:
        p = os.path.join(model_dir, f'lgbm_{mtype}.pkl')
        if os.path.exists(p):
            models[mtype] = joblib.load(p)

    # ─── カテゴリ別シミュレーション ─────────────────────────────
    total = dict(races=0, cost=0, ret=0, bets=0, hits=0)
    category_stats = {m: dict(races=0, cost=0, ret=0, bets=0, hits=0) for m in models}

    for race_id, group in df_2025.groupby('race_id'):
        if len(group) < 5:
            continue
        mtype = group['model_type'].iloc[0]
        if mtype not in models:
            continue

        model = models[mtype]
        grp = group.copy()
        grp['odds_float'] = pd.to_numeric(grp['odds'], errors='coerce').fillna(1.0)

        valid_cols = [c for c in feature_cols if c in grp.columns]
        try:
            grp['pred_prob'] = model.predict_proba(grp[valid_cols])[:, 1]
        except Exception:
            continue

        grp['ev'] = grp['pred_prob'] * grp['odds_float']
        grp = grp.sort_values('pred_prob', ascending=False)

        # 本命(◎)のみ: EVフィルターを通過した場合に購入
        top1 = grp.iloc[0]
        st = category_stats[mtype]
        total['races'] += 1
        st['races'] += 1

        if top1['ev'] >= ev_thresh:
            total['cost'] += 100
            total['bets'] += 1
            st['cost'] += 100
            st['bets'] += 1
            if top1['rank'] == 1.0:
                payout = top1['odds_float'] * 100
                total['ret'] += payout
                total['hits'] += 1
                st['ret'] += payout
                st['hits'] += 1

    # ─── 結果表示 ─────────────────────────────────────────────
    def pct(a, b): return f"{a/b*100:.1f}%" if b > 0 else "N/A"

    print(f"\n{'='*58}")
    print(f"  🏇 専用モデル・実戦シミュレーション 2025年 (EV>={ev_thresh})")
    print(f"{'='*58}")
    print(f"  {'カテゴリ':<12} {'対象R':>5} {'購入':>5} {'的中':>5} {'的中率':>7} {'回収率':>8}")
    print(f"  {'-'*56}")
    for mtype, st in category_stats.items():
        label = {'turf_short':'芝短中距(≤1600)', 'turf_long':'芝長距(≥1800)', 'dirt':'ダート'}[mtype]
        roi  = pct(st['ret'], st['cost'])
        hitr = pct(st['hits'], st['bets'])
        print(f"  {label:<16} {st['races']:>5} {st['bets']:>5} {st['hits']:>5} {hitr:>7} {roi:>8}")
    print(f"  {'-'*56}")
    t = total
    t_roi  = pct(t['ret'], t['cost'])
    t_hitr = pct(t['hits'], t['bets'])
    print(f"  {'合計':<16} {t['races']:>5} {t['bets']:>5} {t['hits']:>5} {t_hitr:>7} {t_roi:>8}")
    print(f"  投資額: {t['cost']:,}円  |  払戻: {int(t['ret']):,}円")
    print(f"{'='*58}\n")

if __name__ == "__main__":
    simulate_specialized(ev_thresh=1.5)
