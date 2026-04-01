"""
Triple Crown 再学習スクリプト - 全期間データで3モデルを更新
2026年実戦投入のため、テスト用に保留していた2025年データも学習に組み込む。
"""
import pandas as pd
import numpy as np
import os, sys, joblib
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
from train_specialized import parse_race_type, assign_model_type
from train_turf_short_final import add_jockey_venue_encoding
from train_dirt_final import add_dirt_specific_features

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')

PARAMS = {
    'turf_short': dict(n_estimators=500, learning_rate=0.02, num_leaves=127,
                       max_depth=7, subsample=0.75, colsample_bytree=0.75,
                       min_child_samples=15, reg_alpha=0.1, reg_lambda=0.5,
                       class_weight='balanced', random_state=42),
    'turf_long':  dict(n_estimators=500, learning_rate=0.02, num_leaves=63,
                       max_depth=6, subsample=0.8, colsample_bytree=0.8,
                       min_child_samples=20, reg_alpha=0.2, reg_lambda=1.0,
                       class_weight='balanced', random_state=42),
    'dirt':       dict(n_estimators=600, learning_rate=0.015, num_leaves=63,
                       max_depth=6, subsample=0.8, colsample_bytree=0.7,
                       min_child_samples=20, reg_alpha=0.15, reg_lambda=0.8,
                       class_weight='balanced', random_state=42),
}

EXCLUDE_BASE = ['rank', 'is_top3', 'race_id', 'year', 'surface', 'distance',
                'model_type', 'race_info', 'odds', 'popularity']
EXCLUDE_CAT  = ['venue_code', 'jockey', 'bracket', 'horse_weight']


def load_and_prepare():
    df  = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'processed', 'model_input.parquet'))
    raw = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'raw', 'scraped_results.parquet'))
    raw = raw[['race_id', 'race_info']].drop_duplicates('race_id')
    df['race_id']  = df['race_id'].astype(str)
    raw['race_id'] = raw['race_id'].astype(str)
    df = df.merge(raw, on='race_id', how='left')

    parsed = df.apply(lambda r: parse_race_type(r['race_id'], r.get('race_info')), axis=1)
    df['surface']    = [p[0] for p in parsed]
    df['distance']   = [p[1] for p in parsed]
    df['model_type'] = df.apply(lambda r: assign_model_type(r['surface'], r['distance']), axis=1)
    df['year']       = df['race_id'].str[:4].astype(int)
    df['is_top3']    = (df['rank'] <= 3).astype(int)
    return df


def train_model(mtype: str, df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  再学習: {mtype.upper()} (全期間データ使用)")
    sub = df[df['model_type'] == mtype].copy()

    # ダート専用特徴量
    if mtype == 'dirt':
        train_mask = pd.Series([True] * len(sub), index=sub.index)
        sub = add_dirt_specific_features(sub, train_mask)
        sub = sub.reset_index(drop=True)

    train_mask = pd.Series([True] * len(sub), index=sub.index)
    sub = add_jockey_venue_encoding(sub, train_mask)
    sub = sub.reset_index(drop=True)

    exc = EXCLUDE_BASE + EXCLUDE_CAT + (['bracket', 'horse_weight'] if mtype == 'dirt' else [])
    if mtype == 'dirt':
        exc += ['bracket', 'horse_weight']
    feature_cols = [c for c in sub.columns if c not in set(exc + ['year'])]

    X = sub[feature_cols]
    y = sub['is_top3']

    print(f"  Total samples: {len(sub)} | Features: {len(feature_cols)}")

    model = lgb.LGBMClassifier(**PARAMS[mtype])
    model.fit(X, y, callbacks=[lgb.log_evaluation(0)])

    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"  Train AUC (overfit check): {auc:.4f}")
    print(f"  Features: {feature_cols[:8]}... ({len(feature_cols)} total)")

    save_path = os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2.pkl')
    meta_path = os.path.join(MODEL_DIR, f'lgbm_{mtype}_v2_meta.pkl')
    joblib.dump(model, save_path)
    joblib.dump({'feature_cols': feature_cols, 'mtype': mtype, 'auc': auc}, meta_path)
    print(f"  Saved => {save_path}")
    return model, feature_cols


def main():
    print("Loading data ...")
    df = load_and_prepare()
    print(f"Total rows: {len(df)}")
    print(f"Years: {sorted(df['year'].unique())}")

    for mtype in ['turf_short', 'turf_long', 'dirt']:
        train_model(mtype, df)

    print(f"\n{'='*55}")
    print("  三冠馬モデル v2 — 全データ学習完了!")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
