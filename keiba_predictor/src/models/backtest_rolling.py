"""
三冠馬 — 完全時系列ローリングバックテスト
==========================================
2025/03/01 〜 2026/03/31

・各レース予測時、そのレース以前のデータのみ使用
・2週間ごとにモデルを再学習（逐次更新）
・EV 1.2 / 1.5 / 2.0 × 単勝 / 複勝(推定) の6パターン
・SHAP値で「万馬券の決め手」と「失敗のノイズ」を解剖
"""
import sys, os, warnings, traceback
warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap as _shap
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

VENUE_TURN = {'01':'L','02':'R','03':'R','04':'L','05':'L',
              '06':'R','07':'L','08':'R','09':'R','10':'R'}

PARAMS = dict(n_estimators=300, learning_rate=0.025, num_leaves=63, max_depth=6,
              subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
              reg_alpha=0.1, reg_lambda=0.5, class_weight='balanced',
              random_state=42, verbose=-1)

EXCLUDE = {
    'rank','is_top3','is_win','race_id','year','surface','distance','model_type',
    'race_info','race_date','race_date_dt','odds','popularity','venue_code',
    'horse_id','horse_name','time','margin','passage_rank','last3f_time',
    'relative_last3f','passage_num','horse_weight','bracket',
    'rank_en','bracket_jp','bracket_en','horse_num_jp','horse_num_en',
    'horse_name_jp','horse_name_en','sex_age_jp','sex_age_en','wc_jp','wc_en',
    'jockey_jp','jockey_en','odds_jp','odds_en','pop_jp','pop_en',
    'hw_jp','hw_en','trainer_jp','trainer_en',
}

EV_THRESHOLDS = [1.2, 1.5, 2.0]
BET_UNIT = 100


# ─── 前処理（完全版）──────────────────────────────────────────
def full_preprocess():
    src = os.path.join(DATA_DIR, 'scraped_results_full.csv')
    if not os.path.exists(src):
        src = os.path.join(DATA_DIR, 'scraped_results.csv')
    print(f"  読込: {src}")
    df = pd.read_csv(src, dtype=str)

    COL_MAP = {'着 順':'rank','枠 番':'bracket','馬 番':'horse_num','馬名':'horse_name',
               '性齢':'sex_age','斤量':'weight_constraint','騎手':'jockey','単勝':'odds',
               '人 気':'popularity','馬体重':'horse_weight','調教師':'trainer'}
    for jp, en in COL_MAP.items():
        if jp in df.columns: df[en] = df[jp]

    for col in ['rank','bracket','horse_num','weight_constraint','odds','popularity']:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

    df['race_id']   = df['race_id'].astype(str).str.strip()
    df['horse_id']  = df.get('horse_id', pd.Series('', index=df.index)).astype(str).str.strip()
    df['race_date'] = df.get('race_date', pd.Series('', index=df.index)).astype(str)
    df['race_info'] = df.get('race_info', pd.Series('', index=df.index)).astype(str)
    df['weather']   = df.get('weather',   pd.Series('晴', index=df.index)).fillna('晴').astype(str)
    df['track']     = df.get('track',     pd.Series('良', index=df.index)).fillna('良').astype(str)
    df['passage_rank'] = df.get('passage_rank', pd.Series('', index=df.index)).astype(str)
    df['last3f_time']  = pd.to_numeric(df.get('last3f_time', pd.Series(dtype=float)), errors='coerce')
    df['horse_weight_raw'] = df['horse_weight'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['race_date_dt'] = pd.to_datetime(df['race_date'], errors='coerce')
    df = df[df['rank'].notna() & (df['rank'] >= 1)].copy()
    df = df.sort_values(['horse_id','race_date_dt']).reset_index(drop=True)

    # 時系列特徴量（Shift → リーク完全回避）
    df['prev_rank']    = df.groupby('horse_id')['rank'].shift(1)
    df['rest_days']    = (df['race_date_dt'] - df.groupby('horse_id')['race_date_dt'].shift(1)).dt.days
    df['weight_delta'] = (df['weight_constraint'] - df.groupby('horse_id')['weight_constraint'].shift(1)).fillna(0)
    df['weight_ratio'] = df['weight_constraint'] / (df['horse_weight_raw'] + 1e-5)

    # 上がり偏差値（レース内のみで計算 → リークゼロ）
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

    # 血統
    ped = os.path.join(DATA_DIR, 'horse_pedigree.csv')
    if os.path.exists(ped):
        p = pd.read_csv(ped, dtype=str); p['horse_id'] = p['horse_id'].str.strip()
        df = df.merge(p[['horse_id','sire','bms','lineage']], on='horse_id', how='left')
    else:
        df['sire'] = 'unknown'; df['bms'] = 'unknown'; df['lineage'] = 'unknown'
    for c in ['sire','bms','lineage']: df[c] = df[c].fillna('unknown').astype(str)
    df['sire_track_interaction'] = df['sire'] + '_' + df['track']
    df['is_long_trip'] = 0; df['is_stay'] = 0

    # 回り順（expanding, shift → リークゼロ）
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

    # レースタイプ
    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info','')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['year']       = df['race_id'].astype(str).str[:4].astype(int)
    df['is_top3']    = (df['rank'] <= 3).astype(int)
    df['is_win']     = (df['rank'] == 1).astype(int)

    print(f"  前処理完了: {len(df):,}行 | 年別: {df['year'].value_counts().sort_index().to_dict()}")
    return df


# ─── モデル学習 ──────────────────────────────────────────────
def fit_models_and_build(df_train):
    les = {}
    for col in ['sex_age','jockey','trainer','weather','track','sire','bms','lineage','sire_track_interaction']:
        if col in df_train.columns:
            le = LabelEncoder(); le.fit(df_train[col].fillna('unknown').astype(str))
            les[col] = le

    def encode(df):
        df = df.copy()
        for col, le in les.items():
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str).apply(
                    lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0)
        return df

    df_enc = encode(df_train)
    trained = {}
    for mtype in ['turf_short','turf_long','dirt']:
        sub = df_enc[df_enc['model_type']==mtype].copy()
        if sub['is_top3'].nunique() < 2 or len(sub) < 30: continue
        if mtype == 'dirt':
            sub = add_dirt_specific_features(sub, pd.Series(True, index=sub.index)).reset_index(drop=True)
        sub = add_jockey_venue_encoding(sub, pd.Series(True, index=sub.index)).reset_index(drop=True)
        feat_cols = [c for c in sub.columns
                     if c not in EXCLUDE and pd.api.types.is_numeric_dtype(sub[c]) and sub[c].notna().any()]
        X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = sub['is_top3']
        m = lgb.LGBMClassifier(**PARAMS); m.fit(X, y, callbacks=[lgb.log_evaluation(0)])
        trained[mtype] = {'model': m, 'feat_cols': feat_cols,
                          'explainer': _shap.TreeExplainer(m)}
    return trained, les, encode


def predict_week(df_week, trained, encode):
    """1週間分のレースを予測 (SHAP計算なし → 後で一括)"""
    results = []
    df_enc = encode(df_week)
    for mtype, obj in trained.items():
        sub = df_enc[df_enc['model_type']==mtype].copy()
        if len(sub) == 0: continue
        if mtype == 'dirt':
            sub = add_dirt_specific_features(sub, pd.Series(True, index=sub.index)).reset_index(drop=True)
        sub = add_jockey_venue_encoding(sub, pd.Series(True, index=sub.index)).reset_index(drop=True)
        feat_cols = obj['feat_cols']
        for c in feat_cols:
            if c not in sub.columns: sub[c] = 0.0
        X = sub[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        sub['pred_prob'] = obj['model'].predict_proba(X)[:,1]
        sub['ev']        = sub['pred_prob'] * sub['odds'].fillna(10)
        sub['mtype']     = mtype
        # 特徴量値を保存（後でSHAP一括計算用）
        for fc in feat_cols[:20]:  # 上位20特徴量のみ
            sub[f'feat_{fc}'] = X[fc].values
        keep = ['race_id','race_date_dt','horse_name','horse_id','rank','is_win','is_top3',
                'odds','pred_prob','ev','mtype','surface','distance','venue_code','turn_dir',
                'weather','track','horse_num','weight_constraint','horse_weight_raw']
        results.append(sub[[c for c in sub.columns if c in keep or c.startswith('feat_')]])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─── 複勝オッズ推定 ─────────────────────────────────────────
def estimate_place_odds(tan_odds, rank):
    """単勝オッズから複勝払戻を粗く推定 (賭式が違うため近似)"""
    if rank > 3: return 0.0
    # 粗い経験則: 複勝 ≈ tan_odds^0.4 × 1.2 (上限8倍、下限1.1倍)
    est = min(max(float(tan_odds) ** 0.4 * 1.2, 1.1), 8.0)
    return round(est * 2) / 2  # 0.5刻みに丸め


# ─── メイン ──────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  🏆 完全時系列ローリングバックテスト 2025/03〜2026/03")
    print("=" * 70)

    df_all = full_preprocess()
    df_all['race_date_dt'] = pd.to_datetime(df_all['race_date_dt'])

    SIM_START = pd.Timestamp('2025-03-01')
    SIM_END   = pd.Timestamp('2026-03-31')
    INIT_END  = SIM_START - timedelta(days=1)   # 初期学習: 〜2025/02/28

    df_init   = df_all[df_all['race_date_dt'] <= INIT_END].copy()
    df_sim    = df_all[(df_all['race_date_dt'] >= SIM_START) &
                       (df_all['race_date_dt'] <= SIM_END)].copy()

    print(f"\n  初期学習データ: {len(df_init):,}行 ({df_init['race_date_dt'].min().date()} 〜 {df_init['race_date_dt'].max().date()})")
    print(f"  シミュレーション: {len(df_sim):,}行 ({SIM_START.date()} 〜 {SIM_END.date()})")

    # 2週間ごとのウィンドウ設定
    window_starts = pd.date_range(SIM_START, SIM_END, freq='2W-SAT')
    print(f"  再学習サイクル: {len(window_starts)}回（2週ごと）\n")

    # 初期モデル学習
    print("  初期モデル学習中...")
    df_accum = df_init.copy()
    trained, les, encode_fn = fit_models_and_build(df_accum)
    print(f"  モデル数: {len(trained)}")

    # 結果格納
    ALL_PREDS   = []   # 全予測行
    SHAP_WINS   = {}   # 的中時のSHAP総和
    SHAP_LOSSES = {}   # 外れ時のSHAP総和
    SHAP_COUNTS = {'win':0, 'loss':0}
    UPDATE_LOG  = []   # 更新ログ

    print(f"\n  {'PERIOD':>20}  {'賭':>5}  {'的中':>5}  {'単勝ROI':>8}  {'更新AUC':>8}")
    print(f"  {'─'*60}")

    for wi, ws in enumerate(window_starts):
        we = ws + timedelta(days=13)  # 2週間ウィンドウ
        df_week = df_sim[(df_sim['race_date_dt'] >= ws) & (df_sim['race_date_dt'] <= we)].copy()
        if len(df_week) == 0: continue

        # 予測
        preds = predict_week(df_week, trained, encode_fn)
        if len(preds) == 0: continue

        # EV ≥ 1.5 のベット
        bets = preds[preds['ev'] >= 1.5].copy()
        n_bets = len(bets); n_wins = int(bets['is_win'].sum()) if n_bets > 0 else 0
        inv = n_bets * BET_UNIT
        ret = int(bets[bets['is_win']==1]['odds'].fillna(0).sum() * BET_UNIT) if n_bets > 0 else 0
        roi = ret/inv*100 if inv > 0 else 0.0

        ALL_PREDS.append(preds)

        # 累積データでモデル更新
        df_accum = df_all[df_all['race_date_dt'] <= we].copy()
        trained, les, encode_fn = fit_models_and_build(df_accum)

        period_str = f"{ws.strftime('%m/%d')}-{we.strftime('%m/%d')}"
        print(f"  {period_str:>20}  {n_bets:>5}  {n_wins:>5}  {roi:>7.1f}%  (更新済)")

    # ─── 全期間集計 ─────────────────────────────────────────
    if not ALL_PREDS:
        print("シミュレーション結果なし"); return

    all_preds = pd.concat(ALL_PREDS, ignore_index=True)

    # ─── SHAP一括計算（最終モデルでサンプルから）─────────────
    print("\n  SHAP分析中（最終モデルで代表サンプル計算）...")
    for mtype, obj in trained.items():
        bets_mt = all_preds[(all_preds['ev']>=1.5)&(all_preds['mtype']==mtype)]
        if len(bets_mt) == 0: continue
        feat_cols = obj['feat_cols']
        # サンプル抽出（最大200件）
        sample = bets_mt.sample(min(200, len(bets_mt)), random_state=42)
        enc_sample = encode_fn(df_all[df_all['race_id'].isin(sample['race_id'])].copy())
        if mtype=='dirt': enc_sample = add_dirt_specific_features(enc_sample, pd.Series(True, index=enc_sample.index)).reset_index(drop=True)
        enc_sample = add_jockey_venue_encoding(enc_sample, pd.Series(True, index=enc_sample.index)).reset_index(drop=True)
        for c in feat_cols:
            if c not in enc_sample.columns: enc_sample[c] = 0.0
        Xs = enc_sample[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        if len(Xs) == 0: continue
        sv = obj['explainer'].shap_values(Xs)
        if isinstance(sv, list): sv = sv[1]
        is_wins = enc_sample['is_win'].values if 'is_win' in enc_sample.columns else np.zeros(len(Xs))
        for j, is_w in enumerate(is_wins):
            key = 'win' if is_w==1 else 'loss'
            SHAP_COUNTS[key] += 1
            target = SHAP_WINS if key=='win' else SHAP_LOSSES
            for fi, feat in enumerate(feat_cols):
                target[feat] = target.get(feat, 0) + abs(sv[j, fi])

    print(f"\n{'='*70}")
    print("  📊 全期間 EV閾値別 回収率サマリー")
    print(f"{'='*70}")
    print(f"  {'EV閾値':>8}  {'件数':>7}  {'単勝的中':>8}  {'単勝ROI':>9}  {'複勝ROI(推定)':>13}")
    print(f"  {'─'*65}")

    summary_rows = []
    for thr in EV_THRESHOLDS:
        bets = all_preds[all_preds['ev'] >= thr].copy()
        if len(bets) == 0: continue
        n = len(bets); wins = int(bets['is_win'].sum()); plc = int(bets['is_top3'].sum())
        inv = n * BET_UNIT

        # 単勝
        ret_tan = int(bets[bets['is_win']==1]['odds'].fillna(0).sum() * BET_UNIT)
        roi_tan = ret_tan/inv*100

        # 複勝（推定）
        bets['place_odds_est'] = bets.apply(
            lambda r: estimate_place_odds(r['odds'], r['rank']), axis=1)
        ret_huku = int(bets[bets['is_top3']==1]['place_odds_est'].fillna(0).sum() * BET_UNIT)
        roi_huku = ret_huku/inv*100

        icon_t = "🟢" if roi_tan>=100 else "🟡" if roi_tan>=90 else "🔴"
        icon_h = "🟢" if roi_huku>=100 else "🟡" if roi_huku>=90 else "🔴"
        print(f"  EV≥{thr:3.1f}  {n:>7,}件  {wins:>5,}({wins/n*100:4.1f}%)  {icon_t}{roi_tan:>7.1f}%  {icon_h}複勝{roi_huku:>6.1f}%(推定)")
        summary_rows.append({'ev_thr':thr,'n':n,'wins':wins,'win_rate':wins/n,
                              'roi_tan':roi_tan,'roi_huku':roi_huku})

    # ─── 月次内訳 ───────────────────────────────────────────
    print(f"\n  【月次収支 — EV≥1.5 単勝】")
    all_preds['ym'] = all_preds['race_date_dt'].dt.to_period('M').astype(str)
    bets15 = all_preds[all_preds['ev'] >= 1.5].copy()
    monthly = bets15.groupby('ym').apply(
        lambda g: pd.Series({'n':len(g), 'wins':g['is_win'].sum(),
                             'inv':len(g)*BET_UNIT,
                             'ret':g[g['is_win']==1]['odds'].fillna(0).sum()*BET_UNIT})).reset_index()
    monthly['roi'] = monthly['ret']/monthly['inv']*100
    monthly['pnl'] = monthly['ret'] - monthly['inv']
    cum = 0
    print(f"  {'月':>8}  {'件数':>5}  {'投資':>8}  {'払戻':>8}  {'損益':>9}  {'ROI':>6}")
    print(f"  {'─'*55}")
    for _, r in monthly.iterrows():
        cum += r['pnl']; ico = "📈" if r['pnl']>=0 else "📉"
        print(f"  {r['ym']:>8}  {int(r['n']):>5}  ¥{int(r['inv']):>6,}  ¥{int(r['ret']):>6,}  "
              f"{'+' if r['pnl']>=0 else ''}¥{int(r['pnl']):>6,}  {r['roi']:>5.0f}% {ico}")
    print(f"  {'累計':>8}  {int(bets15['is_win'].count()):>5}  ¥{int(bets15['is_win'].count()*BET_UNIT):>6,}  "
          f"¥{int(bets15[bets15['is_win']==1]['odds'].fillna(0).sum()*BET_UNIT):>6,}")

    # ─── 穴馬ヒット TOP15 ─────────────────────────────────
    print(f"\n  【万馬券・最高配当 TOP15】")
    wins = all_preds[(all_preds['is_win']==1)&(all_preds['ev']>=1.0)].sort_values('odds',ascending=False)
    print(f"  {'#':>3}  {'馬名':^12}  {'オッズ':>7}  {'EV':>6}  {'日付':>12}  {'条件':^8}")
    print(f"  {'─'*60}")
    for i, (_, r) in enumerate(wins.head(15).iterrows(), 1):
        name = str(r.get('horse_name','-'))[:10]
        odds = float(r.get('odds',0)); ev = float(r.get('ev',0))
        dt   = r['race_date_dt'].strftime('%Y/%m/%d') if pd.notna(r['race_date_dt']) else '-'
        mt   = {'turf_short':'芝短','turf_long':'芝長','dirt':'ダート'}.get(r.get('mtype',''),'-')
        td   = '左' if r.get('turn_dir','')=='L' else '右'
        trk  = str(r.get('track',''))[:2]
        print(f"  {i:>3}  {name:^12}  {odds:>6.1f}倍  {ev:>5.2f}  {dt:>12}  {mt}/{td}/{trk}")

    # ─── 失敗パターン TOP10（高EV外れ） ──────────────────
    print(f"\n  【EV高いのに外れた大失敗 TOP15 — ノイズ検出】")
    fails = all_preds[(all_preds['is_win']==0)&(all_preds['ev']>=2.0)].sort_values('ev',ascending=False)
    print(f"  {'#':>3}  {'馬名':^12}  {'EV':>6}  {'着順':>5}  {'オッズ':>7}  {'条件':^8}  {'日付':>12}")
    print(f"  {'─'*65}")
    for i, (_, r) in enumerate(fails.head(15).iterrows(), 1):
        name = str(r.get('horse_name','-'))[:10]
        ev   = float(r.get('ev',0)); rank = int(r.get('rank',99))
        odds = float(r.get('odds',0))
        dt   = r['race_date_dt'].strftime('%Y/%m/%d') if pd.notna(r['race_date_dt']) else '-'
        mt   = {'turf_short':'芝短','turf_long':'芝長','dirt':'ダート'}.get(r.get('mtype',''),'-')
        td   = '左' if r.get('turn_dir','')=='L' else '右'
        print(f"  {i:>3}  {name:^12}  {ev:>5.2f}  {rank:>4}着  {odds:>6.1f}倍  {mt}/{td}  {dt:>12}")

    # ─── SHAP重要度：ROI貢献 TOP10 ─────────────────────
    print(f"\n{'='*70}")
    print("  🧠 全期間の『真の重要度』— ROI貢献 SHAP TOP10")
    print(f"{'='*70}")
    all_shap = {k: SHAP_WINS.get(k,0)+SHAP_LOSSES.get(k,0) for k in set(list(SHAP_WINS.keys())+list(SHAP_LOSSES.keys()))}
    sorted_all = sorted(all_shap.items(), key=lambda x:x[1], reverse=True)[:10]
    FEAT_JP = {
        'jv_top3_rate':'騎手×競馬場の複勝率','jv_win_rate_adj':'騎手×競馬場勝率(補正)',
        'prev1_relative_last3f':'前走の上がり偏差値','prev2_relative_last3f':'2走前の上がり偏差値',
        'prev3_relative_last3f':'3走前の上がり偏差値','weight_ratio':'斤量/馬体重比率',
        'rest_days':'休養日数','trainer':'調教師スコア','sex_age':'性齢スコア',
        'horse_weight_raw':'馬体重(kg)','horse_num':'馬番','turn_left_rank_avg':'左回り平均着順',
        'turn_right_rank_avg':'右回り平均着順','turn_match_score':'回り適性マッチ度',
        'turn_preference':'左右好み','weight_constraint':'斤量(kg)',
        'j_win_rate':'騎手の総合勝率','bv_win_rate_adj':'枠番×競馬場の勝率',
        'prev1_passage_num':'前走の通過順位','weight_delta':'斤量の増減',
    }
    tot = max(sum(v for _,v in sorted_all), 1)
    print(f"  {'#':>3}  {'特徴量':^28}  {'総SHAP':>8}  {'貢献率':>7}  バー")
    print(f"  {'─'*65}")
    for i, (feat, sv) in enumerate(sorted_all, 1):
        jp  = FEAT_JP.get(feat, feat)[:24]
        pct = sv/tot*100
        bar = '██' * int(pct/3)
        print(f"  {i:>3}  {jp:^28}  {sv:>8.1f}  {pct:>6.1f}%  {bar}")

    # ─── 当たりSHAP vs 外れSHAP ───────────────────────
    print(f"\n  【EV≥1.5 決定打特徴量 — 的中時 vs 外れ時の差分】")
    print(f"  (値が大きい = その予測に多く貢献した特徴量)")
    all_feats = set(list(SHAP_WINS.keys()) + list(SHAP_LOSSES.keys()))
    n_win = max(SHAP_COUNTS['win'], 1); n_loss = max(SHAP_COUNTS['loss'], 1)
    diffs = {}
    for feat in all_feats:
        w = SHAP_WINS.get(feat, 0)/n_win
        l = SHAP_LOSSES.get(feat, 0)/n_loss
        diffs[feat] = {'win_avg':w, 'loss_avg':l, 'diff':w-l}

    sorted_diffs = sorted(diffs.items(), key=lambda x:x[1]['diff'], reverse=True)
    print(f"\n  💎 的中時に効いた特徴量 TOP8（外れ時より寄与が大きい）")
    print(f"  {'特徴量':^28}  {'的中時':>8}  {'外れ時':>8}  {'差分':>8}")
    print(f"  {'─'*58}")
    for feat, vals in sorted_diffs[:8]:
        jp = FEAT_JP.get(feat, feat)[:24]
        print(f"  {jp:^28}  {vals['win_avg']:>8.4f}  {vals['loss_avg']:>8.4f}  {vals['diff']:>+8.4f}")

    print(f"\n  🚫 ノイズだった特徴量 TOP8（外れ時に大きく動いていた）")
    print(f"  {'特徴量':^28}  {'的中時':>8}  {'外れ時':>8}  {'差分':>8}")
    print(f"  {'─'*58}")
    for feat, vals in sorted(sorted_diffs, key=lambda x:x[1]['diff'])[:8]:
        jp = FEAT_JP.get(feat, feat)[:24]
        print(f"  {jp:^28}  {vals['win_avg']:>8.4f}  {vals['loss_avg']:>8.4f}  {vals['diff']:>+8.4f}")

    # ─── 履歴書（最終） ──────────────────────────────────
    print(f"\n{'='*70}")
    print("  📖 AIの履歴書 — 何を信じて勝ち、何を信じて失敗したか")
    print(f"{'='*70}")

    top_win = wins.head(1)
    if len(top_win):
        r = top_win.iloc[0]
        top3_win = sorted(SHAP_WINS.items(), key=lambda x:x[1], reverse=True)[:3]
        w_feats_txt = "、".join([FEAT_JP.get(f,f)[:15] for f,_ in top3_win])
        print(f"\n  🥇 最大的中 — {r.get('horse_name','-')} {float(r.get('odds',0)):.1f}倍")
        print(f"     的中時の決め手: {w_feats_txt}")

    top_fail = fails.head(1)
    if len(top_fail):
        r = top_fail.iloc[0]
        top3_fail_only = sorted(
            [(f,SHAP_LOSSES.get(f,0)/max(SHAP_COUNTS['loss'],1)
              - SHAP_WINS.get(f,0)/max(SHAP_COUNTS['win'],1)) for f in SHAP_LOSSES],
            key=lambda x:x[1], reverse=True)[:3]
        f_feats_txt = "、".join([FEAT_JP.get(f,f)[:15] for f,_ in top3_fail_only])
        print(f"\n  💔 最大失敗 — {r.get('horse_name','-')} EV={float(r.get('ev',0)):.2f}で{int(r.get('rank',0))}着")
        print(f"     ノイズだった特徴量: {f_feats_txt}")

    # ─── 最終結論 ─────────────────────────────────────────
    bets_total = all_preds[all_preds['ev'] >= 1.5]
    inv_total  = len(bets_total)*BET_UNIT
    ret_total  = int(bets_total[bets_total['is_win']==1]['odds'].fillna(0).sum()*BET_UNIT)
    roi_final  = ret_total/inv_total*100 if inv_total > 0 else 0

    print(f"\n{'='*70}")
    print(f"  📋 最終結論（2025/03〜2026/03 完全時系列検証）")
    print(f"  EV≥1.5 単勝戦略: 回収率 {roi_final:.1f}%  {'🟢 黒字' if roi_final>=100 else '🔴 赤字'}")

    # 最良閾値
    best_roi  = 0; best_thr = 1.5
    for thr in [1.0,1.2,1.5,1.8,2.0,2.5,3.0]:
        sub = all_preds[all_preds['ev'] >= thr]
        if len(sub) < 5: continue
        r = sub[sub['is_win']==1]['odds'].fillna(0).sum()*BET_UNIT / (len(sub)*BET_UNIT) * 100
        if r > best_roi: best_roi=r; best_thr=thr

    print(f"  最適EV閾値: {best_thr:.1f}以上 → 回収率 {best_roi:.1f}%")
    print(f"\n  【週末への提言】")
    if best_roi >= 100:
        print(f"  EV≥{best_thr:.1f}の馬に絞れば、この期間では黒字でした。")
        print(f"  4/4・4/5はArゥ: 芝レースでEV≥{best_thr:.1f}の馬に¥500〜¥1,000")
    else:
        print(f"  正直なところ、どの閾値でも赤字でした。")
        print(f"  AIは「参考情報」として活用し、1点¥100〜¥200の楽しめる範囲で。")
    print(f"{'='*70}")

    out_csv = os.path.join(BASE_DIR, 'data', 'rolling_backtest_result.csv')
    feat_cols_drop = [c for c in all_preds.columns if c.startswith('feat_')]
    all_preds.drop(columns=feat_cols_drop, errors='ignore').to_csv(
        out_csv, index=False, encoding='utf-8-sig')
    print(f"\n  全予測結果 → {out_csv}")


if __name__ == '__main__':
    main()
