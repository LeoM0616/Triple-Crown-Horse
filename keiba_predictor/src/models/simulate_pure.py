"""
Pure-skill specialized model simulation.
- オッズ・人気を学習から除外した実力特化モデルで予測
- EV = AI純粋確率(複勝圏内率) × 実際の単勝オッズ  >=  ev_thresh のみ購入
- 券種: 単勝 / ワイドBOX(◎◯▲) / 馬連BOX(◎◯▲)
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib
import math

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type


def estimate_wide_odds(o1: float, o2: float) -> float:
    """単勝オッズから ワイドオッズを推計（実測値がないため）"""
    # 一般的に: wide ≒ (o1 * o2)^0.45 / 2.8
    return max(1.1, (o1 * o2) ** 0.45 / 2.8)


def estimate_quinella_odds(o1: float, o2: float) -> float:
    """単勝オッズから馬連オッズを推計"""
    # 一般的に: quinella ≒ (o1 * o2)^0.5 / 2.0
    return max(1.1, (o1 * o2) ** 0.5 / 2.0)


def simulate_pure(ev_thresh: float = 1.5):
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    proc_pq   = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')
    raw_pq    = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    model_dir = os.path.join(base_dir, 'src', 'models')

    df  = pd.read_parquet(proc_pq)
    raw = pd.read_parquet(raw_pq)[['race_id', 'race_info']].drop_duplicates('race_id')
    df['race_id']  = df['race_id'].astype(str)
    raw['race_id'] = raw['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    # race type
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['year']       = df['race_id'].str[:4].astype(int)
    df_2025 = df[df['year'] >= 2025].copy()

    # 学習と同じ特徴量（オッズ・人気除外）
    ODDS_COLS    = ['odds', 'popularity']
    ignore_cols  = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance',
                    'model_type', 'race_info'] + ODDS_COLS
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # モデルロード
    models = {}
    for mtype in ['turf_short', 'turf_long', 'dirt']:
        p = os.path.join(model_dir, f'lgbm_{mtype}_pure.pkl')
        if os.path.exists(p):
            models[mtype] = joblib.load(p)

    # ─── 集計変数 ──────────────────────────────────────────────
    label = {'turf_short': '芝短中距(≤1600m)', 'turf_long': '芝長距(≥1800m)', 'dirt': 'ダート'}

    stats = {m: dict(
        races=0,
        # 単勝
        win_bets=0, win_hits=0, win_cost=0.0, win_ret=0.0,
        # ワイドBOX
        wide_bets=0, wide_hits=0, wide_cost=0.0, wide_ret=0.0,
        # 馬連BOX
        quin_bets=0, quin_hits=0, quin_cost=0.0, quin_ret=0.0,
    ) for m in models}

    for race_id, group in df_2025.groupby('race_id'):
        if len(group) < 5:
            continue
        mtype = group['model_type'].iloc[0]
        if mtype not in models:
            continue

        grp = group.copy()
        grp['odds_float'] = pd.to_numeric(grp['odds'], errors='coerce').fillna(1.0)

        valid_cols = [c for c in feature_cols if c in grp.columns]
        try:
            grp['pred_prob'] = models[mtype].predict_proba(grp[valid_cols])[:, 1]
        except Exception:
            continue

        # EV = 純粋実力確率(複勝率相当) × 単勝オッズ
        grp['ev'] = grp['pred_prob'] * grp['odds_float']
        grp = grp.sort_values('pred_prob', ascending=False)

        st = stats[mtype]
        st['races'] += 1

        # ─── 単勝: 本命(◎)の ev >= threshold なら購入 ─────────
        top1 = grp.iloc[0]
        if top1['ev'] >= ev_thresh:
            st['win_cost'] += 100
            st['win_bets'] += 1
            if top1['rank'] == 1.0:
                st['win_hits'] += 1
                st['win_ret'] += top1['odds_float'] * 100

        # ─── ワイドBOX + 馬連BOX: 確率上位3頭(◎◯▲) ───────────
        if len(grp) >= 3:
            picks = grp.head(3)
            pairs = [(picks.iloc[i], picks.iloc[j]) for i in range(3) for j in range(i+1, 3)]

            for a, b in pairs:
                oa, ob = a['odds_float'], b['odds_float']
                ra, rb = a['rank'], b['rank']

                # ワイドBOX: 両馬が3着以内に入れば的中
                w_odds = estimate_wide_odds(oa, ob)
                st['wide_cost'] += 100
                st['wide_bets'] += 1
                if ra <= 3.0 and rb <= 3.0:
                    st['wide_hits'] += 1
                    st['wide_ret'] += w_odds * 100

                # 馬連BOX: 両馬が1・2着（順不同）
                q_odds = estimate_quinella_odds(oa, ob)
                st['quin_cost'] += 100
                st['quin_bets'] += 1
                if ra <= 2.0 and rb <= 2.0:
                    st['quin_hits'] += 1
                    st['quin_ret'] += q_odds * 100

    # ─── 表示 ───────────────────────────────────────────────────
    def pct(a, b):
        return f"{a/b*100:.1f}%" if b > 0 else "-"

    def roi(ret, cost):
        return f"{ret/cost*100:.1f}%" if cost > 0 else "-"

    print(f"\n{'='*72}")
    print(f"  🏇 純粋実力3モデル 実戦シミュレーション 2025年  (EV>={ev_thresh})")
    print(f"  ※ オッズ・人気を学習から除外した純粋実力ベースの予測確率を使用")
    print(f"{'='*72}")

    totals = dict(races=0, win_bets=0, win_hits=0, win_cost=0.0, win_ret=0.0,
                  wide_bets=0, wide_hits=0, wide_cost=0.0, wide_ret=0.0,
                  quin_bets=0, quin_hits=0, quin_cost=0.0, quin_ret=0.0)

    for mtype, st in stats.items():
        for k in totals:
            totals[k] += st.get(k, 0)

        print(f"\n  ▶ {label[mtype]}  ({st['races']}レース)")
        print(f"  {'券種':<10} {'購入':>5} {'的中':>5} {'的中率':>7} {'回収率':>8}  {'払戻':>10} / {'投資':>10}")
        print(f"  {'-'*66}")
        print(f"  {'単勝':<10} {st['win_bets']:>5} {st['win_hits']:>5} {pct(st['win_hits'],st['win_bets']):>7} {roi(st['win_ret'],st['win_cost']):>8}  {int(st['win_ret']):>10,}円 / {int(st['win_cost']):>8,}円")
        print(f"  {'ワイドBOX':<10} {st['wide_bets']:>5} {st['wide_hits']:>5} {pct(st['wide_hits'],st['wide_bets']):>7} {roi(st['wide_ret'],st['wide_cost']):>8}  {int(st['wide_ret']):>10,}円 / {int(st['wide_cost']):>8,}円  ※推計")
        print(f"  {'馬連BOX':<10} {st['quin_bets']:>5} {st['quin_hits']:>5} {pct(st['quin_hits'],st['quin_bets']):>7} {roi(st['quin_ret'],st['quin_cost']):>8}  {int(st['quin_ret']):>10,}円 / {int(st['quin_cost']):>8,}円  ※推計")

    t = totals
    print(f"\n  {'─'*66}")
    print(f"  ★ 合計 ({t['races']}レース)")
    print(f"  {'単勝':<10} {t['win_bets']:>5} {t['win_hits']:>5} {pct(t['win_hits'],t['win_bets']):>7} {roi(t['win_ret'],t['win_cost']):>8}  {int(t['win_ret']):>10,}円 / {int(t['win_cost']):>8,}円")
    print(f"  {'ワイドBOX':<10} {t['wide_bets']:>5} {t['wide_hits']:>5} {pct(t['wide_hits'],t['wide_bets']):>7} {roi(t['wide_ret'],t['wide_cost']):>8}  {int(t['wide_ret']):>10,}円 / {int(t['wide_cost']):>8,}円  ※推計")
    print(f"  {'馬連BOX':<10} {t['quin_bets']:>5} {t['quin_hits']:>5} {pct(t['quin_hits'],t['quin_bets']):>7} {roi(t['quin_ret'],t['quin_cost']):>8}  {int(t['quin_ret']):>10,}円 / {int(t['quin_cost']):>8,}円  ※推計")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    simulate_pure(ev_thresh=1.5)
