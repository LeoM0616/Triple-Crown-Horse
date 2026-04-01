import pandas as pd
import os

# プロジェクトのルートディレクトリからCSVファイルのパスを指定
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(base_dir, 'data', 'raw', 'sample_race_data.csv')

def main():
    print(f"Loading data from: {csv_path}\n")
    
    # CSVファイルをデータフレームとして読み込み
    df = pd.read_csv(csv_path)
    
    print("=== 先頭のデータ (head) ===")
    print(df.head())
    print("\n")
    
    print("=== データの基本情報 (info) ===")
    df.info()
    print("\n")
    
    print("=== 基本統計量 (describe) ===")
    print(df.describe())

if __name__ == "__main__":
    main()
