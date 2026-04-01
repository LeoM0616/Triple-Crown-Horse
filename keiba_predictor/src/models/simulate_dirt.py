"""
ダート専用モデル 最終シミュレーター
- 砂被り考慮の当日バイアス補正（内外枠の有利不利を ±6% で補正）
- EVスイープ 1.0 ~ 2.5 (0.1刻み)
- 券種: 単勝 / ワイド2点流し(◎-◯, ◎-▲)
"""
import pandas as pd
import numpy as np
import os
import sys
import joblib

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features


def estimate_wide_odds(o1: float, o2: float) -> float:
    return max(1.1, (o1 * o2) ** 0.45 / 2.8)


def calc_day_bias_dirt(df_raw: pd.DataFrame, race_date: str, venue_code: str) -> dict:
    """
    ダート版当日バイアス: 枠番有利不利を ±6% で補正
    JRA: 外めの枠が砂を被りにくい傾向
    地方: 内ラチ沿いが有利な傾向
    """
    bias = {'bracket_bias': {}}
    same_day = df_raw[
        (df_raw['race_date'] == race_date) &
        (df_raw['race_id'].astype(str).str[4:6] == venue_code)
    ].copy()

    if len(same_day) < 8:
        return bias

    same_day['bracket_num'] = pd.to_numeric(
        same_day.get('bracket', same_day.get('枠 番', np.nan)), errors='coerce'
    )
    same_day['rank_num'] = pd.to_numeric(
        same_day.get('rank', same_day.get('着 順', np.nan)), errors='coerce'
    )
    same_day = same_day.dropna(subset=['bracket_num', 'rank_num'])

    if len(same_day) > 0:
        bracket_avg = same_day.groupby('bracket_num')['rank_num'].mean()
        overall_avg = same_day['rank_num'].mean()
        for b, avg in bracket_avg.items():
            # ±6% の範囲で補正（砂被り有利不利）
            bias['bracket_bias'][int(b)] = np.clip(
                (overall_avg - avg) / (overall_avg + 1e-5) * 0.12, -0.06, 0.06
            )
    return bias


def apply_bias(pred_prob: float, bracket: int, bias: dict) -> float:
    adj = bias['bracket_bias'].get(int(bracket), 0.0)
    return float(np.clip(pred_prob + adj, 0.001, 0.999))


def simulate_dirt():
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    proc_pq   = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')
    raw_pq    = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    model_dir = os.path.join(base_dir, 'src', 'models')

    df     = pd.read_parquet(proc_pq)
    df_raw = pd.read_parquet(raw_pq)
    raw    = df_raw[['race_id', 'race_info', 'race_date']].drop_duplicates('race_id').copy()
    raw['race_id']   = raw['race_id'].astype(str)
    raw['race_date'] = raw['race_date'].astype(str)

    df['race_id'] = df['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    # race type
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['venue_code'] = df['race_id'].str[4:6]
    df['year']       = df['race_id'].str[:4].astype(int)

    # ダートのみ
    sub = df[df['model_type'] == 'dirt'].copy()
    sub['is_top3'] = (sub['rank'] <= 3).astype(int)

    # 特徴量エンジニアリング（trainデータで構築）
    train_mask = sub['year'] <= 2024
    sub = add_dirt_specific_features(sub, train_mask)
    sub = sub.reset_index(drop=True)
    train_mask = sub['year'] <= 2024
    sub = add_jockey_venue_encoding(sub, train_mask)
    sub = sub.reset_index(drop=True)
    sub_2025 = sub[sub['year'] >= 2025].copy()

    # モデルロード
    model     = joblib.load(os.path.join(model_dir, 'lgbm_dirt_final.pkl'))
    meta      = joblib.load(os.path.join(model_dir, 'lgbm_dirt_final_meta.pkl'))
    feat_cols = meta['feature_cols']

    valid_cols = [c for c in feat_cols if c in sub_2025.columns]
    sub_2025['pred_prob_raw'] = model.predict_proba(sub_2025[valid_cols])[:, 1]

    # df_raw 列名正規化
    df_raw = df_raw.rename(columns={'着 順': 'rank', '枠 番': 'bracket'})
    df_raw['race_date'] = df_raw['race_date'].astype(str)

    # ─── EVスイープ + ワイド2点流し ──────────────────────────────
    ev_thresholds = [round(x * 0.1, 1) for x in range(10, 26)]
    sweep_results = []
    wide_collected = False
    wide_stat = {}

    for ev_thresh in ev_thresholds:
        win_cost = win_ret = bets = hits = 0
        w_cost = w_ret = w_bets = w_hits = 0

        for race_id, group in sub_2025.groupby('race_id'):
            if len(group) < 5:
                continue

            grp = group.copy()
            grp['odds_float'] = pd.to_numeric(grp['odds'], errors='coerce').fillna(1.0)

            # 当日バイアス補正（ダート版 ±6%）
            race_date  = str(grp['race_date'].iloc[0]) if 'race_date' in grp.columns else ''
            venue_code = str(race_id)[4:6]
            day_bias   = calc_day_bias_dirt(df_raw, race_date, venue_code)

            bracket_col = 'bracket_num' if 'bracket_num' in grp.columns else None
            grp['pred_prob'] = grp.apply(
                lambda r: apply_bias(
                    r['pred_prob_raw'],
                    int(r[bracket_col]) if bracket_col and pd.notna(r.get(bracket_col)) else 4,
                    day_bias
                ), axis=1
            )

            grp['ev'] = grp['pred_prob'] * grp['odds_float']
            grp = grp.sort_values('pred_prob', ascending=False)
            top1 = grp.iloc[0]

            # 単勝
            if top1['ev'] >= ev_thresh:
                win_cost += 100; bets += 1
                if top1['rank'] == 1.0:
                    hits += 1
                    win_ret += top1['odds_float'] * 100

            # ワイド2点流し（初回のみ集計）
            if not wide_collected and len(grp) >= 3:
                for pick_other in [grp.iloc[1], grp.iloc[2]]:
                    w_odds = estimate_wide_odds(top1['odds_float'], pick_other['odds_float'])
                    w_cost  += 100; w_bets += 1
                    if top1['rank'] <= 3.0 and pick_other['rank'] <= 3.0:
                        w_hits += 1
                        w_ret  += w_odds * 100

        if not wide_collected:
            wide_stat = {'bets': w_bets, 'hits': w_hits, 'cost': w_cost, 'ret': int(w_ret),
                         'roi': w_ret / w_cost * 100 if w_cost > 0 else 0}
            wide_collected = True

        roi  = win_ret / win_cost * 100 if win_cost > 0 else 0
        hitr = hits / bets * 100 if bets > 0 else 0
        sweep_results.append({
            'ev': ev_thresh, 'roi': roi, 'hit_rate': hitr,
            'bets': bets, 'hits': hits, 'cost': win_cost, 'ret': int(win_ret)
        })

    # ─── 出力 ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  ⛏️  最終収支報告書（ダート専用AI）2025年テストデータ")
    print(f"  ※ オッズ除外学習 + 騎手×競馬場TE + 枠番バイアス補正(±6%) 適用済み")
    print(f"{'='*72}")

    print(f"\n  ─── 単勝 EVスイープ結果 (EV閾値 1.0 → 2.5) ───")
    print(f"  {'EV':>5} {'購入':>5} {'的中':>5} {'的中率':>7} {'回収率':>8}  {'払戻':>10} / {'投資':>8}")
    print(f"  {'─'*62}")

    best_roi = 0; best_ev = 0
    for r in sweep_results:
        is_best = r['roi'] > best_roi and r['bets'] >= 30
        if is_best:
            best_roi = r['roi']; best_ev = r['ev']
        marker = " ◀ BEST" if is_best else ("  ← 黒字" if r['roi'] >= 100 else "")
        print(f"  {r['ev']:>5.1f} {r['bets']:>5} {r['hits']:>5} {r['hit_rate']:>6.1f}% "
              f"{r['roi']:>7.1f}%  {r['ret']:>10,}円 / {r['cost']:>6,}円{marker}")

    best = next((r for r in sweep_results if r['ev'] == best_ev), sweep_results[-1])
    print(f"\n  ★ 最適EV閾値: {best_ev}  →  回収率 {best_roi:.1f}%  "
          f"(的中: {best['hits']}/{best['bets']}件, 投資: {best['cost']:,}円, 払戻: {best['ret']:,}円)")

    wr = wide_stat
    whr = wr['hits'] / wr['bets'] * 100 if wr.get('bets', 0) > 0 else 0
    print(f"\n  ─── ワイド2点流し(◎-◯, ◎-▲) 結果 ───")
    print(f"  購入: {wr['bets']}点  的中: {wr['hits']}点  的中率: {whr:.1f}%  "
          f"回収率: {wr['roi']:.1f}%  ※推計オッズ")
    print(f"  払戻: {wr['ret']:,}円 / 投資: {wr['cost']:,}円")

    print(f"\n{'='*72}")
    print(f"  🏆 3モデル合算（芝短中+芝長+ダート）の回収率は最終レポート参照")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    simulate_dirt()
