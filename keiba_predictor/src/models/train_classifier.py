import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'processed', 'model_input.csv')
    model_dir = os.path.join(base_dir, 'src', 'models')
    model_path = os.path.join(model_dir, 'lgbm_classifier_walkforward.pkl')
    
    print(f"Loading preprocessed data from {input_path}")
    df = pd.read_csv(input_path, dtype={'race_id': str})
    
    # 目的変数(y): 3着以内を1、それ以外を0とする二値分類
    target_col = 'is_top3'
    df[target_col] = (df['rank'] <= 3).astype(int)
    
    # 時系列分割用に年を抽出
    df['year'] = df['race_id'].astype(str).str[:4].astype(int)
    
    ignore_cols = ['rank', target_col, 'race_id', 'year']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    # 学習用（<=2024）とテスト用（>=2025: 完全に未知の未来）に分割
    train_df = df[df['year'] <= 2024]
    test_df = df[df['year'] >= 2025]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Train data (<=2024): {len(train_df)} rows")
    print(f"Test data (>=2025): {len(test_df)} rows")
    
    print("Training LightGBM Classifier (Walk-Forward) ...")
    # 不均衡データ (上位3頭 vs その他) のため、is_unbalance=True または scale_pos_weight 等を指定
    # 今回はシンプルに確率を出したいため class_weight='balanced' を利用
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 予測と評価
    preds_proba = model.predict_proba(X_test)[:, 1] # 3着以内に入る確率
    preds_binary = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds_binary)
    auc = roc_auc_score(y_test, preds_proba)
    loss = log_loss(y_test, preds_proba)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC AUC: {auc:.4f} (1.0に近いほど優秀)")
    print(f"Test Log Loss: {loss:.4f} (小さいほど優秀)")
    
    # 特徴量重要度の確認
    importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importance.head(10))
    
    # モデルの保存
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nClassifier model saved to: {model_path}")

if __name__ == '__main__':
    main()
