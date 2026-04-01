"""
三冠馬 (Triple Crown) - 統合予測エンジン
======================================================
2023-2025年の全データで再学習した3専用モデルを統合し、
race_idから条件を自動判別して予想を出力する最終システム。

特徴:
  - 自動モデルセレクター（芝短・芝長・ダート）
  - SHAP値で個体別の選択根拠を説明
  - EV上位3頭を◎◯▲形式で出力
  - プロフェッショナルな収支報告書フォーマット
"""
import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
import shap
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features

# ─── 特徴量の日本語説明マップ ──────────────────────────────────
FEAT_JP = {
    'jv_top3_rate':           '騎手×競馬場の複勝率',
    'jv_win_rate':            '騎手×競馬場の勝率',
    'jv_win_rate_adj':        '騎手×競馬場の勝率(補正)',
    'jv_top2_rate':           '騎手×競馬場の連対率',
    'j_top3_rate':            '騎手の総合複勝率',
    'j_win_rate':             '騎手の総合勝率',
    'v_win_rate':             '競馬場の平均勝率',
    'prev1_relative_last3f':  '前走の上がり偏差値',
    'prev2_relative_last3f':  '前々走の上がり偏差値',
    'prev3_relative_last3f':  '3走前の上がり偏差値',
    'prev_rank':              '前走着順',
    'prev1_passage_num':      '前走の通過順位',
    'prev2_passage_num':      '前々走の通過順位',
    'prev3_passage_num':      '3走前の通過順位',
    'weight_ratio':           '斤量/馬体重比率',
    'weight_delta':           '斤量の増減(前走比)',
    'horse_weight_raw':       '馬体重',
    'weight_constraint':      '斤量',
    'rest_days':              '休養日数',
    'sire_track_interaction': '父系統×馬場状態',
    'bv_win_rate_adj':        '枠番×競馬場の勝率',
    'dirt_bracket_score':     '枠番砂被りスコア',
    'is_inner_bracket':       '内枠フラグ',
    'is_outer_bracket':       '外枠フラグ',
    'weight_burden_score':    '斤量負担スコア',
    'sex_age':                '性別・年齢',
    'horse_num':              '馬番',
    'sire':                   '父名',
    'bms':                    '母父名',
    'lineage':                '血統系統',
    'weather':                '天候',
    'track':                  '馬場状態',
    'bracket':                '枠番',
    'bracket_num':            '枠番',
    'is_long_trip':           '長距離輸送フラグ',
    'is_stay':                '滞在フラグ',
}


class TripleCrown:
    """三冠馬統合予測エンジン"""

    MODEL_DIR = os.path.join(os.path.dirname(__file__))

    def __init__(self):
        print("🏆 TRIPLE CROWN ENGINE 起動中...")
        self.models = {}
        self.metas  = {}
        self.explainers = {}

        for mtype in ['turf_short', 'turf_long', 'dirt']:
            mp = os.path.join(self.MODEL_DIR, f'lgbm_{mtype}_v2.pkl')
            ep = os.path.join(self.MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl')
            if os.path.exists(mp):
                self.models[mtype] = joblib.load(mp)
                self.metas[mtype]  = joblib.load(ep)
                self.explainers[mtype] = shap.TreeExplainer(self.models[mtype])
                print(f"  ✓ {mtype:<12} モデル読込完了 (features={len(self.metas[mtype]['feature_cols'])})")
            else:
                print(f"  ✗ {mtype} モデルが見つかりません: {mp}")

        # 学習データから JV-TE 統計を構築
        self._build_jv_stats()
        print("  ✓ JV統計 (騎手×競馬場) 読込完了")
        print("🏆 TRIPLE CROWN ENGINE 準備完了!\n")

    def _build_jv_stats(self):
        """全学習データから騎手×競馬場の勝率統計を構築"""
        base = os.path.abspath(os.path.join(self.MODEL_DIR, '..', '..'))
        df = pd.read_parquet(os.path.join(base, 'data', 'processed', 'model_input.parquet'))
        df['race_id'] = df['race_id'].astype(str)
        df['venue_code'] = df['race_id'].str[4:6]
        df['is_win']  = (df['rank'] == 1.0).astype(float)
        df['is_top2'] = (df['rank'] <= 2.0).astype(float)
        df['is_top3'] = (df['rank'] <= 3.0).astype(float)

        # 全データで jv 統計
        self._jv_stats = df.groupby(['jockey', 'venue_code']).agg(
            jv_win_rate  =('is_win',  'mean'),
            jv_top2_rate =('is_top2', 'mean'),
            jv_top3_rate =('is_top3', 'mean'),
            jv_race_count=('is_win',  'count'),
        ).reset_index()
        self._j_stats = df.groupby('jockey').agg(
            j_win_rate  =('is_win',  'mean'),
            j_top3_rate =('is_top3', 'mean'),
        ).reset_index()
        self._v_stats = df.groupby('venue_code').agg(
            v_win_rate  =('is_win',  'mean'),
            v_top3_rate =('is_top3', 'mean'),
        ).reset_index()
        self._global_win  = df['is_win'].mean()
        self._global_top3 = df['is_top3'].mean()

        # 枠番×競馬場 統計（ダート用）
        self._bv_stats = df.groupby(['venue_code', 'bracket']).agg(
            bv_win_rate=('is_win', 'mean'),
            bv_count   =('is_win', 'count'),
        ).reset_index()
        self._bv_stats['bv_win_rate_adj'] = (
            self._bv_stats['bv_win_rate'] * self._bv_stats['bv_count'] + self._global_win * 5
        ) / (self._bv_stats['bv_count'] + 5)

    def _enrich_entry(self, entry: dict, race_meta: dict) -> dict:
        """出走馬エントリーに JV-TE と競馬場コードを付与"""
        e = dict(entry)
        race_id    = race_meta['race_id']
        venue_code = str(race_id)[4:6]
        jockey     = e.get('jockey', '')

        # JV統計のマージ
        jv_row = self._jv_stats[
            (self._jv_stats['jockey'] == jockey) &
            (self._jv_stats['venue_code'] == venue_code)
        ]
        j_row = self._j_stats[self._j_stats['jockey'] == jockey]
        v_row = self._v_stats[self._v_stats['venue_code'] == venue_code]

        k = 10
        if len(jv_row) > 0:
            rc = jv_row.iloc[0]['jv_race_count']
            e['jv_win_rate']     = jv_row.iloc[0]['jv_win_rate']
            e['jv_top2_rate']    = jv_row.iloc[0]['jv_top2_rate']
            e['jv_top3_rate']    = jv_row.iloc[0]['jv_top3_rate']
            e['jv_race_count']   = rc
            e['jv_win_rate_adj'] = (e['jv_win_rate'] * rc + self._global_win * k) / (rc + k)
        else:
            jwr = j_row.iloc[0]['j_win_rate'] if len(j_row) > 0 else self._global_win
            jt3 = j_row.iloc[0]['j_top3_rate'] if len(j_row) > 0 else self._global_top3
            e['jv_win_rate']     = jwr
            e['jv_top2_rate']    = jt3 * 0.6
            e['jv_top3_rate']    = jt3
            e['jv_race_count']   = 0
            e['jv_win_rate_adj'] = (jwr * 0 + self._global_win * k) / k

        e['j_win_rate']  = j_row.iloc[0]['j_win_rate']  if len(j_row) > 0 else self._global_win
        e['j_top3_rate'] = j_row.iloc[0]['j_top3_rate'] if len(j_row) > 0 else self._global_top3
        e['v_win_rate']  = v_row.iloc[0]['v_win_rate']  if len(v_row) > 0 else self._global_win

        # 枠番×競馬場（ダート用）
        bracket_num = int(e.get('bracket', 4))
        bv_row = self._bv_stats[
            (self._bv_stats['venue_code'] == venue_code) &
            (self._bv_stats['bracket'].astype(str) == str(bracket_num))
        ]
        e['bv_win_rate_adj'] = bv_row.iloc[0]['bv_win_rate_adj'] if len(bv_row) > 0 else self._global_win

        # ダート専用特徴量
        hw_str = str(e.get('horse_weight', '480'))
        hw = float(''.join(c for c in hw_str.split('(')[0] if c.isdigit() or c == '.') or '480')
        e['horse_weight_raw']    = hw
        e['weight_burden_score'] = e.get('weight_constraint', 57) / (hw + 1e-5)
        e['is_inner_bracket']    = 1 if bracket_num <= 2 else 0
        e['is_outer_bracket']    = 1 if bracket_num >= 7 else 0
        is_local = 1 if int(venue_code) >= 30 else 0
        e['is_local_dirt']       = is_local
        e['dirt_bracket_score']  = (3 - bracket_num) * 0.02 if is_local else (bracket_num - 4.5) * 0.015
        e['bracket_num']         = bracket_num

        return e

    def predict_race(self, race_id: str, race_meta: dict) -> pd.DataFrame:
        """1レースを予測してDataFrameを返す"""
        surface  = race_meta['surface']
        distance = race_meta['distance']
        mtype    = assign_model_type(surface, distance)
        race_meta['race_id'] = race_id

        if mtype not in self.models:
            print(f"  [WARN] {mtype} モデルが未ロードです")
            return pd.DataFrame()

        model     = self.models[mtype]
        feat_cols = self.metas[mtype]['feature_cols']
        explainer = self.explainers[mtype]

        rows = []
        for entry in race_meta['entries']:
            e = self._enrich_entry(entry, race_meta)
            rows.append(e)

        df = pd.DataFrame(rows)

        # 欠損列を0埋め
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0

        X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        df['pred_prob']   = model.predict_proba(X)[:, 1]
        df['odds_float']  = pd.to_numeric(df['odds'], errors='coerce').fillna(1.0)
        df['ev']          = df['pred_prob'] * df['odds_float']
        df['mtype']       = mtype
        df['race_id_key'] = race_id

        # SHAP値を計算
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # クラス1 (top3入り) のSHAP値

        df['_shap_top5'] = [
            sorted(zip(feat_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
            for sv in shap_vals
        ]

        return df.sort_values('ev', ascending=False).reset_index(drop=True)

    def format_prediction(self, race_id: str, race_meta: dict) -> str:
        """予測結果をプロ仕様のレポート形式で返す"""
        df     = self.predict_race(race_id, race_meta)
        if df.empty:
            return ""

        mtype   = df['mtype'].iloc[0]
        surface = race_meta['surface']
        dist    = race_meta['distance']
        suf_jp  = '芝' if surface == 'turf' else 'ダ'
        cat_jp  = {'turf_short':'芝・短中距離専用モデル',
                   'turf_long': '芝・中長距離専用モデル',
                   'dirt':      'ダート専用モデル'}[mtype]

        marks  = ['◎', '◯', '▲']
        lines  = []
        lines.append(f"\n{'═'*62}")
        lines.append(f"  {race_meta['name']}")
        lines.append(f"  {race_meta['date']}  {race_meta['venue']}  {suf_jp}{dist}m")
        lines.append(f"  使用モデル: {cat_jp}")
        lines.append(f"{'═'*62}")

        for i, (_, row) in enumerate(df.head(3).iterrows()):
            mark  = marks[i]
            name  = row.get('horse_name', f"#{row['horse_num']}")
            prob  = row['pred_prob'] * 100
            ev    = row['ev']
            odds  = row['odds_float']
            jock  = row.get('jockey', '-')
            sire  = row.get('sire', '-')

            lines.append(f"\n  {mark} 馬番{int(row['horse_num']):2d}  {name}")
            lines.append(f"     騎手: {jock}  |  単勝オッズ: {odds:.1f}倍  "
                         f"|  実力確率: {prob:.1f}%  |  ★EV: {ev:.2f}")

            # SHAP Top5 説明
            lines.append(f"     【AI選出根拠 TOP5】")
            for feat, sv in row['_shap_top5']:
                feat_val = row.get(feat, 0)
                jp       = FEAT_JP.get(feat, feat)
                direction = "+" if sv > 0 else "-"
                try:
                    val_str = f"{float(feat_val):.3f}"
                except (TypeError, ValueError):
                    val_str = str(feat_val)[:12]
                lines.append(f"       {direction} {jp:<22} = {val_str:<12}  (SHAP: {sv:+.3f})")

            lines.append(f"     父: {sire}  |  前走着順: {int(row.get('prev_rank',0))}着  "
                         f"|  斤量: {row.get('weight_constraint',0)}kg")

        # 買い目サマリー
        top3 = df.head(3)
        lines.append(f"\n  {'─'*58}")
        lines.append(f"  📋 推奨買い目")
        ev1 = top3.iloc[0]['ev']
        o1, o2, o3 = top3.iloc[0]['odds_float'], top3.iloc[1]['odds_float'], top3.iloc[2]['odds_float']
        n1, n2, n3 = int(top3.iloc[0]['horse_num']), int(top3.iloc[1]['horse_num']), int(top3.iloc[2]['horse_num'])
        if ev1 >= 1.5:
            lines.append(f"  単勝: {n1}番  EV={ev1:.2f} {'★黒字期待!' if ev1>=1.8 else ''}")
        lines.append(f"  ワイド2点流し: {n1}番→{n2}番 / {n1}番→{n3}番")
        lines.append(f"{'═'*62}")

        return '\n'.join(lines)


def main():
    from race_entries_apr_2026 import ALL_RACES

    engine = TripleCrown()

    print("=" * 62)
    print("  🏇 2026年4月4日・5日 ターゲットレース 最終予想")
    print("  三冠馬 (Triple Crown) v2.0 — 2025年までの全データで学習済み")
    print("=" * 62)

    for race_id, race_meta in ALL_RACES.items():
        report = engine.format_prediction(race_id, race_meta)
        print(report)


if __name__ == '__main__':
    main()
