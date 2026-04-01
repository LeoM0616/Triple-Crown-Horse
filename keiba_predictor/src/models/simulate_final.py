"""
最終決戦シミュレーター：芝・短中距離専用モデル
1. 当日バイアス補正（枠番・通過順位の有利不利をスコア化して補正）
2. EVスイープ 1.0 ~ 2.5 (0.1刻み)
3. 券種: 単勝 / ワイド2点流し(◎-◯, ◎-▲)
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


# ─── 当日バイアス補正ロジック ─────────────────────────────────
def calc_day_bias(df_raw: pd.DataFrame, race_date: str, venue_code: str) -> dict:
    """
    同じ開催日・同じ競馬場の過去のレース（1R〜9R相当）から
    枠番と通過順位の有利不利を計算してバイアス辞書を返す。
    """
    bias = {'bracket_bias': {}, 'passage_bias': 0.0}

    # 当日・同会場のレースを取得
    same_day = df_raw[
        (df_raw['race_date'] == race_date) &
        (df_raw['race_id'].astype(str).str[4:6] == venue_code)
    ].copy()

    if len(same_day) < 10:
        return bias  # データ不足の場合は補正なし

    # 枠番バイアス: 枠番ごとの平均着順（小さいほど優秀）→ 平均との差でスコア化
    same_day['bracket_num'] = pd.to_numeric(same_day.get('bracket', same_day.get('枠 番', np.nan)), errors='coerce')
    same_day['rank_num']    = pd.to_numeric(same_day.get('rank', same_day.get('着 順', np.nan)), errors='coerce')
    same_day = same_day.dropna(subset=['bracket_num', 'rank_num'])

    if len(same_day) > 0:
        bracket_avg = same_day.groupby('bracket_num')['rank_num'].mean()
        overall_avg = same_day['rank_num'].mean()
        # 平均より良い枠 → プラス補正、悪い枠 → マイナス補正
        # スコアを ±0.05 (5%) の範囲に収める
        for b, avg in bracket_avg.items():
            bias['bracket_bias'][int(b)] = np.clip((overall_avg - avg) / (overall_avg + 1e-5) * 0.10, -0.05, 0.05)

    return bias


def apply_bias(pred_prob: float, bracket: int, bias: dict) -> float:
    """予測確率に当日バイアス補正を適用"""
    b_adj = bias['bracket_bias'].get(int(bracket), 0.0)
    adjusted = pred_prob + b_adj
    return float(np.clip(adjusted, 0.001, 0.999))


# ─── オッズ推計関数 ──────────────────────────────────────────
def estimate_wide_odds(o1: float, o2: float) -> float:
    return max(1.1, (o1 * o2) ** 0.45 / 2.8)


# ─── メインシミュレーション ──────────────────────────────────
def simulate_final():
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    proc_pq   = os.path.join(base_dir, 'data', 'processed', 'model_input.parquet')
    raw_pq    = os.path.join(base_dir, 'data', 'raw', 'scraped_results.parquet')
    model_dir = os.path.join(base_dir, 'src', 'models')

    # データ読み込み
    df     = pd.read_parquet(proc_pq)
    df_raw = pd.read_parquet(raw_pq)
    raw    = df_raw[['race_id', 'race_info', 'race_date']].drop_duplicates('race_id').copy()

    for c in ['race_id', 'race_info', 'race_date']:
        raw[c] = raw[c].astype(str) if raw[c].dtype != 'object' else raw[c]

    df['race_id']  = df['race_id'].astype(str)
    raw['race_id'] = raw['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    # race type
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['venue_code'] = df['race_id'].str[4:6]
    df['year']       = df['race_id'].str[:4].astype(int)

    # 芝短中距のみ絞り込み
    sub = df[df['model_type'] == 'turf_short'].copy()
    sub['is_top3'] = (sub['rank'] <= 3).astype(int)

    # ターゲットエンコーディング（train側のデータで構築してtestに適用）
    train_mask_bool = sub['year'] <= 2024
    sub = add_jockey_venue_encoding(sub, train_mask_bool)
    sub_2025 = sub[sub['year'] >= 2025].copy()

    # モデルとメタデータのロード
    model     = joblib.load(os.path.join(model_dir, 'lgbm_turf_short_final.pkl'))
    meta      = joblib.load(os.path.join(model_dir, 'lgbm_turf_short_final_meta.pkl'))
    feat_cols = meta['feature_cols']

    valid_cols = [c for c in feat_cols if c in sub_2025.columns]
    sub_2025['pred_prob_raw'] = model.predict_proba(sub_2025[valid_cols])[:, 1]

    # df_raw の列名正規化
    df_raw = df_raw.rename(columns={
        '着 順': 'rank', '枠 番': 'bracket', '馬 番': 'horse_num'
    })
    df_raw['race_date'] = df_raw['race_date'].astype(str)

    # ─── EV スイープ結果格納用 ────────────────────────────────
    ev_thresholds = [round(x * 0.1, 1) for x in range(10, 26)]  # 1.0 ~ 2.5

    sweep_results = []
    wide_results  = []

    for ev_thresh in ev_thresholds:
        win_cost = win_ret = bets = hits = 0
        w_cost   = w_ret   = w_bets = w_hits = 0

        for race_id, group in sub_2025.groupby('race_id'):
            if len(group) < 5:
                continue

            grp = group.copy()
            grp['odds_float'] = pd.to_numeric(grp['odds'], errors='coerce').fillna(1.0)

            # 当日バイアス補正
            race_date  = str(grp['race_date'].iloc[0]) if 'race_date' in grp.columns else ''
            venue_code = str(race_id)[4:6]
            day_bias   = calc_day_bias(df_raw, race_date, venue_code)

            bracket_col = 'bracket' if 'bracket' in grp.columns else None
            grp['pred_prob'] = grp.apply(
                lambda r: apply_bias(
                    r['pred_prob_raw'],
                    int(r[bracket_col]) if bracket_col and pd.notna(r[bracket_col]) else 0,
                    day_bias
                ), axis=1
            )

            # EV計算 (AIの純粋確率 × 実際のオッズ)
            grp['ev'] = grp['pred_prob'] * grp['odds_float']
            grp = grp.sort_values('pred_prob', ascending=False)
            top1 = grp.iloc[0]

            # 単勝: 本命(◎)がEV閾値を超えたら購入
            if top1['ev'] >= ev_thresh:
                win_cost += 100; bets += 1
                if top1['rank'] == 1.0:
                    hits += 1
                    win_ret += top1['odds_float'] * 100

            # ワイド2点流し (◎-◯, ◎-▲) EVフィルターなしで常に実施
            if ev_thresh == ev_thresholds[0] and len(grp) >= 3:
                pick1 = grp.iloc[0]
                for pick_other in [grp.iloc[1], grp.iloc[2]]:
                    wide_odds = estimate_wide_odds(pick1['odds_float'], pick_other['odds_float'])
                    w_cost  += 100; w_bets += 1
                    if pick1['rank'] <= 3.0 and pick_other['rank'] <= 3.0:
                        w_hits += 1
                        w_ret  += wide_odds * 100

        roi  = win_ret  / win_cost  * 100 if win_cost  > 0 else 0
        wroi = w_ret    / w_cost    * 100 if w_cost    > 0 else 0
        sweep_results.append({'ev': ev_thresh, 'roi': roi, 'hit_rate': hits/bets*100 if bets>0 else 0,
                               'bets': bets, 'hits': hits, 'cost': win_cost, 'ret': int(win_ret)})
        if ev_thresh == ev_thresholds[0]:
            wide_results.append({'w_roi': wroi, 'w_bets': w_bets, 'w_hits': w_hits,
                                  'w_cost': w_cost, 'w_ret': int(w_ret)})

    wr = wide_results[0] if wide_results else {}

    # ─── 最終レポート出力 ─────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  🏆 最終収支報告書 ― 芝・短中距離専用AI (2025年 完全未知データ)")
    print(f"  ※ オッズ除外学習 + 騎手×競馬場TE + 当日バイアス補正 適用済み")
    print(f"{'='*72}")

    # EVスイープ表
    print(f"\n  ─── 単勝 EVスイープ結果 (EV閾値 1.0 → 2.5) ───")
    print(f"  {'EV':>5} {'購入':>5} {'的中':>5} {'的中率':>7} {'回収率':>8}  {'払戻':>10} / {'投資':>8}")
    print(f"  {'─'*60}")
    best_roi = 0
    best_ev  = 0
    for r in sweep_results:
        marker = ""
        if r['roi'] > best_roi and r['bets'] >= 20:
            best_roi = r['roi']
            best_ev  = r['ev']
            marker = " ◀ BEST"
        print(f"  {r['ev']:>5.1f} {r['bets']:>5} {r['hits']:>5} {r['hit_rate']:>6.1f}% {r['roi']:>7.1f}%  "
              f"{r['ret']:>10,}円 / {r['cost']:>6,}円{marker}")

    sr = next((r for r in sweep_results if r['ev'] == best_ev), sweep_results[0])
    print(f"\n  ★ 最適EV閾値: {best_ev}  →  回収率 {best_roi:.1f}%  "
          f"(的中: {sr['hits']}/{sr['bets']}件, 投資: {sr['cost']:,}円, 払戻: {sr['ret']:,}円)")

    # ワイド2点流し
    if wr:
        print(f"\n  ─── ワイド2点流し(◎-◯, ◎-▲) 結果（全レース対象）───")
        print(f"  購入: {wr['w_bets']}点  的中: {wr['w_hits']}点  "
              f"的中率: {wr['w_hits']/wr['w_bets']*100:.1f}%  "
              f"回収率: {wr['w_roi']:.1f}%  ※推計オッズ")
        print(f"  払戻: {wr['w_ret']:,}円 / 投資: {wr['w_cost']:,}円")

    print(f"\n{'='*72}")
    print(f"  ✅ JRAの控除率(約80%の壁)を突破できた場合、それは本物のエッジです！")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    simulate_final()
