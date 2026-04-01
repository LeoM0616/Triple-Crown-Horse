# 競馬予想ツール (Keiba Predictor)

過去のレースデータを解析し、競馬の予想を行うためのツールです。

## ディレクトリ構成
- `data/`
  - `raw/`: 収集した生データ（CSVなど）を配置します。サンプルで `sample_race_data.csv` を用意しています。
  - `processed/`: 前処理済みのデータを配置します。
- `notebooks/`: データ探索・分析（EDA）用のJupyter Notebookを配置します。
- `src/`: ソースコード
  - `data/`: データ収集のスクリプト（スクレイピング等）
  - `features/`: 特徴量エンジニアリングのスクリプト
  - `models/`: 機械学習モデルの学習・予測スクリプト
- `requirements.txt`: 必要なPythonパッケージの一覧です。
