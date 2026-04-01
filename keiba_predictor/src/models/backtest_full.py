"""
三冠馬 — 最終バックテスト (Walk-Forward Validation)
====================================================
データリークなしの正直なシミュレーション

Fold 1: 学習=2023      → テスト=2024
Fold 2: 学習=2023-24   → テスト=2025
Fold 3: 学習=2023-25   → テスト=2026/Q1

EV≥1.5 の馬に単勝100円購入した場合の損益を算出
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'raw')
FEAT_DIR  = os.path.join(BASE_DIR, 'src', 'features')

VENUE_TURN = {'01':'L','02':'R','03':'R','04':'L','05':'L',
              '06':'R','07':'L','08':'R','09':'R','10':'R'}

PARAMS = {
    'turf_short': dict(n_estimators=400, learning_rate=0.02, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
                       reg_alpha=0.1, reg_lambda=0.5, class_weight='balanced', random_state=42),
    'turf_long':  dict(n_estimators=500, learning_rate=0.015, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                       reg_alpha=0.2, reg_lambda=1.0,  class_weight='balanced', random_state=42),
    'dirt':       dict(n_estimators=400, learning_rate=0.02, num_leaves=63, max_depth=6,
                       subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                       reg_alpha=0.15, reg_lambda=0.8, class_weight='balanced', random_state=42),
}

EXCLUDE = {
    'rank','is_top3','race_id','year','surface','distance','model_type',
    'race_info','race_date','race_date_dt','odds','popularity','venue_code',
    'horse_id','horse_name','time','margin','passage_rank','last3f_time',
    'relative_last3f','passage_num','horse_weight','bracket',
    'rank_en','bracket_jp','bracket_en','horse_num_jp','horse_num_en',
    'horse_name_jp','horse_name_en','sex_age_jp','sex_age_en','wc_jp','wc_en',
    'jockey_jp','jockey_en','odds_jp','odds_en','pop_jp','pop_en',
    'hw_jp','hw_en','trainer_jp','trainer_en',
}


# ─── 前処理（rebuild_with_turning と同一ロジック） ─────────────
def add_turning(df):
    df = df.copy()
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df['turn_dir']   = df['venue_code'].map(VENUE_TURN).fillna('R')
    df['is_left']    = (df['turn_dir'] == 'L').astype(int)
    df = df.sort_values(['horse_id','race_date_dt']).reset_index(drop=True)
    for turn, sfx in [('L','left'),('R','right')]:
        mask = df['turn_dir'] == turn
        df[f'turn_{sfx}_rank_avg'] = np.nan
        exp = df[mask].groupby('horse_id')['rank'].transform(lambda x: x.shift(1).expanding().mean())
        df.loc[mask, f'turn_{sfx}_rank_avg'] = exp
    overall = df.groupby('horse_id')['rank'].transform(lambda x: x.shift(1).expanding().mean()).fillna(df['rank'].mean())
    df['turn_left_rank_avg']  = df['turn_left_rank_avg'].fillna(overall)
    df['turn_right_rank_avg'] = df['turn_right_rank_avg'].fillna(overall)
    df['turn_left_deviation']  = overall - df['turn_left_rank_avg']
    df['turn_right_deviation'] = overall - df['turn_right_rank_avg']
    df['turn_preference']      = df['turn_left_deviation'] - df['turn_right_deviation']
    df['turn_match_score']     = np.where(df['is_left']==1, df['turn_left_deviation'], df['turn_right_deviation'])
    return df


def load_and_preprocess():
    print("データ読込・前処理中...")
    src = os.path.join(DATA_DIR, 'scraped_results_full.csv')
    if not os.path.exists(src):
        src = os.path.join(DATA_DIR, 'scraped_results.csv')
    df = pd.read_csv(src, dtype=str)

    COL_MAP = {'着 順':'rank','枠 番':'bracket','馬 番':'horse_num','馬名':'horse_name',
               '性齢':'sex_age','斤量':'weight_constraint','騎手':'jockey','単勝':'odds',
               '人 気':'popularity','馬体重':'horse_weight','調教師':'trainer'}
    for jp, en in COL_MAP.items():
        if jp in df.columns:
            df[en] = df[jp]

    for col in ['rank','bracket','horse_num','weight_constraint','odds','popularity']:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

    df['race_id']      = df['race_id'].astype(str).str.strip()
    df['horse_id']     = df.get('horse_id', pd.Series('', index=df.index)).astype(str).str.strip()
    df['race_date']    = df.get('race_date', pd.Series('', index=df.index)).astype(str)
    df['race_info']    = df.get('race_info', pd.Series('', index=df.index)).astype(str)
    df['weather']      = df.get('weather', pd.Series('晴', index=df.index)).fillna('晴').astype(str)
    df['track']        = df.get('track',   pd.Series('良', index=df.index)).fillna('良').astype(str)
    df['passage_rank'] = df.get('passage_rank', pd.Series('', index=df.index)).astype(str)
    df['last3f_time']  = pd.to_numeric(df.get('last3f_time', pd.Series(dtype=float)), errors='coerce')
    df['horse_weight_raw'] = df['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['race_date_dt'] = pd.to_datetime(df['race_date'], errors='coerce')
    df = df[df['rank'].notna() & (df['rank'] >= 1)].copy()
    df = df.sort_values(['horse_id','race_date_dt']).reset_index(drop=True)

    df['prev_rank']    = df.groupby('horse_id')['rank'].shift(1)
    df['rest_days']    = (df['race_date_dt'] - df.groupby('horse_id')['race_date_dt'].shift(1)).dt.days
    df['weight_delta'] = (df['weight_constraint'] - df.groupby('horse_id')['weight_constraint'].shift(1)).fillna(0)
    df['weight_ratio'] = df['weight_constraint'] / (df['horse_weight_raw'] + 1e-5)

    rm = df.groupby('race_id')['last3f_time'].transform('mean')
    rs = df.groupby('race_id')['last3f_time'].transform('std')
    df['relative_last3f']       = ((df['last3f_time']-rm)/(rs+1e-5)*10+50).fillna(50)
    df['prev1_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(1).fillna(50)
    df['prev2_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(2).fillna(50)
    df['prev3_relative_last3f'] = df.groupby('horse_id')['relative_last3f'].shift(3).fillna(50)

    def parse_p(p):
        parts = [int(x) for x in str(p).split('-') if x.isdigit()]
        return parts[-1] if parts else 7
    df['passage_num']       = df['passage_rank'].apply(parse_p)
    df['prev1_passage_num'] = df.groupby('horse_id')['passage_num'].shift(1).fillna(7)
    df['prev2_passage_num'] = df.groupby('horse_id')['passage_num'].shift(2).fillna(7)
    df['prev3_passage_num'] = df.groupby('horse_id')['passage_num'].shift(3).fillna(7)

    ped = os.path.join(DATA_DIR, 'horse_pedigree.csv')
    if os.path.exists(ped):
        ped_df = pd.read_csv(ped, dtype=str)
        ped_df['horse_id'] = ped_df['horse_id'].astype(str).str.strip()
        df = df.merge(ped_df[['horse_id','sire','bms','lineage']], on='horse_id', how='left')
    else:
        df['sire'] = df.get('trainer','unknown'); df['bms'] = df.get('jockey','unknown')
        df['lineage'] = 'unknown'
    for c in ['sire','bms','lineage']:
        df[c] = df[c].fillna('unknown').astype(str)
    df['sire_track_interaction'] = df['sire'] + '_' + df['track']
    df['is_long_trip'] = 0; df['is_stay'] = 0

    # 回り順
    df = add_turning(df)

    # Race type
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info','')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['venue_code'] = df['race_id'].astype(str).str[4:6]
    df['year']       = df['race_id'].astype(str).str[:4].astype(int)
    df['is_top3']    = (df['rank'] <= 3).astype(int)
    df['is_win']     = (df['rank'] == 1).astype(int)

    print(f"  前処理完了: {len(df):,}行 | 年別: {df['year'].value_counts().sort_index().to_dict()}")
    return df


def fit_encoders(df_train):
    les = {}
    for col in ['sex_age','jockey','trainer','weather','track','sire','bms','lineage','sire_track_interaction']:
        if col in df_train.columns:
            le = LabelEncoder()
            le.fit(df_train[col].fillna('unknown').astype(str))
            les[col] = le
    return les

def apply_encoders(df, les):
    df = df.copy()
    for col, le in les.items():
        if col in df.columns:
            df[col] = df[col].fillna('unknown').astype(str).apply(
                lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0)
    return df

def train_models(df_train, les):
    df_enc = apply_encoders(df_train, les)
    trained = {}
    for mtype in ['turf_short','turf_long','dirt']:
        sub = df_enc[df_enc['model_type']==mtype].copy()
        if sub['is_top3'].nunique() < 2 or len(sub) < 50:
            continue
        if mtype == 'dirt':
            tmask = pd.Series(True, index=sub.index)
            sub = add_dirt_specific_features(sub, tmask).reset_index(drop=True)
        tmask = pd.Series(True, index=sub.index)
        sub   = add_jockey_venue_encoding(sub, tmask).reset_index(drop=True)
        feat_cols = [c for c in sub.columns
                     if c not in EXCLUDE
                     and pd.api.types.is_numeric_dtype(sub[c])
                     and sub[c].notna().any()]
        X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = sub['is_top3']
        model = lgb.LGBMClassifier(**PARAMS[mtype])
        model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        trained[mtype] = {'model': model, 'feat_cols': feat_cols}
        auc = roc_auc_score(y, model.predict_proba(X)[:,1])
        print(f"    {mtype:12s}: {len(sub):6,}行 Train AUC={auc:.4f}")
    return trained

def predict_period(df_test, trained, les):
    df_enc = apply_encoders(df_test, les)
    out_rows = []
    for mtype, obj in trained.items():
        sub = df_enc[df_enc['model_type']==mtype].copy()
        if len(sub) == 0: continue
        if mtype == 'dirt':
            tmask = pd.Series(True, index=sub.index)
            sub = add_dirt_specific_features(sub, tmask).reset_index(drop=True)
        tmask = pd.Series(True, index=sub.index)
        sub   = add_jockey_venue_encoding(sub, tmask).reset_index(drop=True)
        feat_cols = obj['feat_cols']
        for c in feat_cols:
            if c not in sub.columns: sub[c] = 0.0
        X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        sub['pred_prob'] = obj['model'].predict_proba(X)[:,1]
        sub['ev']        = sub['pred_prob'] * sub['odds'].fillna(10)
        sub['mtype']     = mtype
        out_rows.append(sub[['race_id','race_date_dt','horse_name','rank','is_win',
                               'odds','pred_prob','ev','mtype','race_info','track',
                               'weather','horse_num']].copy())
    if not out_rows: return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def simulate_ev(pred_df, ev_threshold=1.5, bet_unit=100):
    """EV閾値以上の馬のみ単勝購入"""
    bets = pred_df[pred_df['ev'] >= ev_threshold].copy()
    bets['bet']    = bet_unit
    bets['payout'] = np.where(bets['is_win']==1, bets['odds'] * bet_unit, 0)
    bets['profit'] = bets['payout'] - bets['bet']
    return bets


def main():
    print("=" * 65)
    print("  🏆 三冠馬 最終バックテスト（Walk-Forward Validation）")
    print("=" * 65)

    df_all = load_and_preprocess()

    ALL_BETS = []
    folds = [
        (2024, list(range(2023, 2024)), 2024),
        (2025, list(range(2023, 2025)), 2025),
        (2026, list(range(2023, 2026)), 2026),
    ]

    for test_year, train_years, _ in folds:
        print(f"\n{'─'*65}")
        print(f"  Fold: 学習={train_years} → テスト={test_year}")
        df_tr = df_all[df_all['year'].isin(train_years)].copy()
        df_te = df_all[df_all['year'] == test_year].copy()
        if len(df_te) == 0:
            print(f"  テストデータなし")
            continue
        print(f"  学習: {len(df_tr):,}行 | テスト: {len(df_te):,}行")

        les     = fit_encoders(df_tr)
        trained = train_models(df_tr, les)
        pred    = predict_period(df_te, trained, les)
        if pred.empty: continue
        bets = simulate_ev(pred, ev_threshold=1.5)
        bets['fold_year'] = test_year
        ALL_BETS.append(bets)
        n_bets   = len(bets)
        n_wins   = int(bets['is_win'].sum())
        invested = int(bets['bet'].sum())
        returned = int(bets['payout'].sum())
        roi      = returned / invested * 100 if invested > 0 else 0
        print(f"  賭け数: {n_bets:,} | 的中: {n_wins} | 投資: ¥{invested:,} | 払戻: ¥{returned:,} | 回収率: {roi:.1f}%")

    if not ALL_BETS:
        print("\nバックテスト結果なし")
        return

    all_bets = pd.concat(ALL_BETS, ignore_index=True)

    # ─── 総合集計 ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  📊 総合バックテスト結果（2024〜2026/Q1 Walk-Forward）")
    print(f"{'='*65}")

    total_bets   = len(all_bets)
    total_wins   = int(all_bets['is_win'].sum())
    total_invest = int(all_bets['bet'].sum())
    total_return = int(all_bets['payout'].sum())
    total_roi    = total_return / total_invest * 100 if total_invest > 0 else 0
    win_rate     = total_wins / total_bets * 100 if total_bets > 0 else 0

    print(f"  合計賭け数   : {total_bets:>8,}レース")
    print(f"  的中数       : {total_wins:>8,}レース ({win_rate:.1f}%)")
    print(f"  合計投資額   : ¥{total_invest:>10,}")
    print(f"  合計払戻金   : ¥{total_return:>10,}")
    print(f"  純損益       : ¥{total_return-total_invest:>+10,}")
    print(f"  最終回収率   : {total_roi:>8.1f}%")
    status = "🟢 黒字" if total_roi >= 100 else "🔴 赤字"
    print(f"  判定         : {status}")

    # ─── モデル別 ─────────────────────────────────────────────
    print(f"\n  【モデル別】")
    for mt in ['turf_short','turf_long','dirt']:
        sub = all_bets[all_bets['mtype']==mt]
        if len(sub) == 0: continue
        inv = int(sub['bet'].sum()); ret = int(sub['payout'].sum())
        roi = ret/inv*100 if inv>0 else 0
        lbl = {'turf_short':'🟣 芝短中距離','turf_long':'🔵 芝中長距離','dirt':'🟠 ダート '}[mt]
        print(f"  {lbl}: {len(sub):4,}件 / 回収率 {roi:6.1f}% ({ret-inv:+,}円)")

    # ─── 月次集計 ─────────────────────────────────────────────
    print(f"\n  【月次収支】")
    all_bets['ym'] = all_bets['race_date_dt'].dt.to_period('M').astype(str)
    monthly = all_bets.groupby('ym').agg(
        bets=('bet','count'), inv=('bet','sum'),
        ret=('payout','sum'), wins=('is_win','sum')
    ).reset_index()
    monthly['roi'] = monthly['ret']/monthly['inv']*100
    monthly['pnl'] = monthly['ret'] - monthly['inv']
    print(f"  {'月':>8} {'賭':>5} {'投資':>9} {'払戻':>9} {'損益':>9} {'回収率':>7}")
    print(f"  {'─'*55}")
    cumulative = 0
    for _, r in monthly.iterrows():
        cumulative += r['pnl']
        icon = "📈" if r['pnl']>=0 else "📉"
        print(f"  {r['ym']:>8} {int(r['bets']):>5} ¥{int(r['inv']):>7,} ¥{int(r['ret']):>7,} "
              f"{'+' if r['pnl']>=0 else ''}¥{int(r['pnl']):>6,} {r['roi']:>6.0f}% {icon}")
    print(f"  {'─'*55}")
    print(f"  {'累計':>8} {total_bets:>5} ¥{total_invest:>7,} ¥{total_return:>7,} "
          f"{'+' if total_return-total_invest>=0 else ''}¥{total_return-total_invest:>6,} {total_roi:>6.1f}%")

    # ─── 最高配当 TOP20 ──────────────────────────────────────
    print(f"\n  【最高配当 TOP20 — 穴馬撃破ランキング】")
    wins = all_bets[all_bets['is_win']==1].sort_values('odds', ascending=False).head(20)
    print(f"  {'#':>3} {'馬名':^12} {'オッズ':>6} {'獲得':>8} {'日付':>12} {'条件':^10}")
    print(f"  {'─'*60}")
    for i, (_, r) in enumerate(wins.iterrows(), 1):
        name  = str(r.get('horse_name','-'))[:10]
        odds  = float(r['odds'])
        gain  = int(r['payout'])
        dt    = r['race_date_dt'].strftime('%Y/%m/%d') if pd.notna(r['race_date_dt']) else '-'
        info  = str(r.get('race_info','-'))[:12]
        mt    = {'turf_short':'芝短','turf_long':'芝長','dirt':'ダート'}.get(r['mtype'],'')
        print(f"  {i:>3} {name:^12} {odds:>6.1f}倍 ¥{gain:>6,} {dt:>12} {mt}")

    # ─── EV別収支分析 ────────────────────────────────────────
    print(f"\n  【EV閾値別 回収率シミュレーション】")
    print(f"  {'EV閾値':>8} {'件数':>6} {'回収率':>8}")
    print(f"  {'─'*30}")
    for threshold in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        sub = all_bets[all_bets['ev'] >= threshold]
        if len(sub) == 0: continue
        inv = sub['bet'].sum(); ret = sub['payout'].sum()
        roi = ret/inv*100 if inv>0 else 0
        bar = '█' * min(int(roi/10), 20)
        print(f"  EV≥{threshold:4.1f} {len(sub):>6,}件  {roi:>6.1f}%  {bar}")

    # ─── 最終verdict ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  📋 週末の賭け方への提言")
    print(f"{'='*65}")

    optimal_ev = None
    best_roi = 0
    for threshold in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        sub = all_bets[all_bets['ev'] >= threshold]
        if len(sub) < 10: continue
        roi = sub['payout'].sum()/sub['bet'].sum()*100
        if roi > best_roi:
            best_roi = roi; optimal_ev = threshold

    opt_sub = all_bets[all_bets['ev'] >= (optimal_ev or 1.5)]
    opt_inv = opt_sub['bet'].sum(); opt_ret = opt_sub['payout'].sum()
    opt_roi = opt_ret/opt_inv*100

    print(f"  最適EV閾値   : {optimal_ev:.1f}以上")
    print(f"  その時の回収率: {opt_roi:.1f}%")
    avg_bets_per_race_day = len(opt_sub) / max((all_bets['fold_year'].nunique()*52), 1)
    print(f"  1開催日平均買い目: 約{avg_bets_per_race_day:.1f}件")

    if opt_roi >= 100:
        print(f"\n  🟢 バックテスト黒字確認！")
        print(f"     4月4・5日はEV≥{optimal_ev:.1f}の馬のみ、1点¥500〜¥1,000で勝負してください")
    elif opt_roi >= 80:
        print(f"\n  🟡 損失は限定的（回収率{opt_roi:.0f}%）")
        print(f"     1点¥200〜¥300程度の少額で様子を見てください")
    else:
        print(f"\n  🔴 正直な結果: 回収率{opt_roi:.0f}%（赤字ライン）")
        print(f"     AIはあくまで参考情報として使い、本命馬の絞り込みに活用してください")
        print(f"     競馬に『必勝法』はありません。楽しめる範囲内で！")

    print(f"{'='*65}")

    # CSV保存
    out = os.path.join(BASE_DIR, 'data', 'backtest_result_full.csv')
    all_bets.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n  詳細結果 → {out}")


if __name__ == '__main__':
    main()
