import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_path = os.path.join(base_dir, 'data', 'processed', 'model_input.csv')
    model_dir = os.path.join(base_dir, 'src', 'models')
    model_path = os.path.join(model_dir, 'lgbm_model.pkl')
    
    print(f"Loading preprocessed data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 特徴量(X)と目的変数(y)
    # 'rank' を予測する。'race_id' は学習には使わない。
    target_col = 'rank'
    ignore_cols = [target_col, 'race_id']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 学習用とテスト用に分割 (今回はランダム分割。実際の競馬推論なら日付順による時系列分割がベター)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training LightGBM Regressor ...")
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 予測と評価
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE (Root Mean Squared Error): {rmse:.4f}")
    
    # 特徴量重要度の確認
    importance = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    print("\nFeature Importances:")
    print(importance)
    
    # モデルの保存
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == '__main__':
    main()
