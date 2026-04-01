"""EV threshold / probability grid search to find optimal betting parameters."""
import pandas as pd
import numpy as np
import os
import joblib
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def run_sim(df_2025, model, feature_cols, ev_thresh, prob_thresh):
    df_2025 = df_2025.copy()
    df_2025['pred_prob'] = model.predict_proba(df_2025[feature_cols])[:, 1]

    win_cost = win_return = bets = hits = 0
    for _, group in df_2025.groupby('race_id'):
        if len(group) < 5:
            continue
        grp = group.copy()
        grp['odds_float'] = pd.to_numeric(grp['odds'], errors='coerce').fillna(1.0)
        grp['ev'] = grp['pred_prob'] * grp['odds_float']
        # 全頭からEVフィルター（穴馬発掘: 確率が高い + オッズが旨い馬を狙う）
        targets = grp[(grp['pred_prob'] >= prob_thresh) & (grp['ev'] >= ev_thresh)]
        for _, horse in targets.iterrows():
            win_cost += 100
            bets += 1
            if horse['rank'] == 1.0:
                hits += 1
                win_return += horse['odds_float'] * 100

    roi = (win_return / win_cost * 100) if win_cost > 0 else 0
    hit = (hits / bets * 100) if bets > 0 else 0
    return roi, hit, bets, hits

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path  = os.path.join(base_dir, 'data', 'processed', 'model_input.csv')
    model_path  = os.path.join(base_dir, 'src', 'models', 'lgbm_classifier_walkforward.pkl')

    df   = pd.read_csv(input_path, dtype={'race_id': str})
    model = joblib.load(model_path)

    df['year'] = df['race_id'].str[:4].astype(int)
    df_2025 = df[df['year'] >= 2025].copy()

    ignore_cols = ['rank', 'is_top3', 'race_id', 'year']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    print(f"\n{'EV':>5} {'P_min':>6} | {'ROI':>7} {'Hit%':>6} {'Bets':>6} {'Wins':>5}")
    print("-" * 48)
    best_roi, best_cfg = 0, {}
    for ev in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        for pmin in [0.10, 0.15, 0.20, 0.25, 0.30]:
            roi, hit, bets, wins = run_sim(df_2025, model, feature_cols, ev, pmin)
            marker = " <-- BEST" if roi > best_roi and bets > 50 else ""
            if roi > best_roi and bets > 50:
                best_roi = roi
                best_cfg = dict(ev=ev, pmin=pmin, roi=roi, hit=hit, bets=bets, wins=wins)
            print(f"{ev:>5.1f} {pmin:>6.2f} | {roi:>6.1f}% {hit:>5.1f}% {bets:>6} {wins:>5}{marker}")

    print("=" * 48)
    print(f"BEST: EV>={best_cfg.get('ev')} / P>={best_cfg.get('pmin')} -> ROI {best_cfg.get('roi'):.1f}%  Hit {best_cfg.get('hit'):.1f}%  ({best_cfg.get('wins')}/{best_cfg.get('bets')})")

if __name__ == "__main__":
    main()
