import pandas as pd
import numpy as np
import os
import joblib
import sys

# Windows環境でのUnicodeEncodeError対策
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_mark(prob, rank):
    """ 指定された確率順位に応じた競馬新聞的な印を返す """
    if rank == 1: return "◎ (本命)"
    if rank == 2: return "◯ (対抗)"
    if rank == 3: return "▲ (単穴)"
    if rank in [4, 5]: return "△ (連下)"
    if rank == 6: return "☆ (穴)"
    return "－"

def generate_newspaper_artifact(result_df):
    """
    AI予想の結果をArtifactとして表示するためのMarkdownを生成・保存する。
    """
    board_md = "# 🏇 AI競馬新聞『AIKeiba』予想ボード\n\n"
    board_md += "## 【今週末の注目G1】 2026年 大阪杯 (阪神・芝2000m)\n\n"
    board_md += "> [!TIP]\n> **AIの注目ポイント**\n> 過去数万件のレース傾向と「上がり偏差値」「脚質」を元に、『3着以内に入る確率（複勝圏内率）』を算出しました。◎本命は驚きの高確率をマーク！\n\n"
    
    board_md += "| 予想印 | 馬番 | 馬名 | AI複勝確率 | 単勝予想オッズ | 脚質予測 | 上がり3F偏差値 |\n"
    board_md += "| :---: | :---: | :--- | :---: | :---: | :---: | :---: |\n"
    
    for i, row in result_df.iterrows():
        mark = row['AI印'].split(' ')[0] # ◎だけ取る
        prob = f"{row['複勝確率(%)']}%"
        # 確率が高いものは太字に
        if row['AI予想Rank'] <= 3:
            prob = f"**{prob}**"
            mark = f"**{mark}**"
            
        board_md += f"| {mark} | {row['馬番']} | {row['馬名']} | {prob} | {row['オッズ']} | {row['脚質']} | {row['上がり偏差値']} |\n"
        
    board_md += "\n"
    board_md += "### 🤖 AIの買い目推奨\n"
    board_md += f"- **単勝**: {result_df.iloc[0]['馬番']} {result_df.iloc[0]['馬名']}\n"
    board_md += f"- **馬連**: {result_df.iloc[0]['馬番']} - {result_df.iloc[1]['馬番']}, {result_df.iloc[0]['馬番']} - {result_df.iloc[2]['馬番']}\n"
    board_md += f"- **3連複**: {result_df.iloc[0]['馬番']} - {result_df.iloc[1]['馬番']} - {result_df.iloc[2]['馬番']}\n"

    # Artifactとして保存
    artifact_path = os.path.join(base_dir, 'artifacts', 'ai_racing_newspaper.md')
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write(board_md)
    print(f"\nArtifact generated: {artifact_path}")


def predict_future_race():
    print(f"\n--- 🏇 未来レース予想 (2026年 大阪杯) ---\n")
    
    # 手動モック/スクレイピングした出馬表の読み込み
    shutuba_path = os.path.join(base_dir, 'data', 'raw', 'osaka_hai_2026_shutuba.csv')
    df_race = pd.read_csv(shutuba_path)
    
    # エンコーダの適用
    encoders_path = os.path.join(base_dir, 'src', 'features', 'encoders.pkl')
    encoders = joblib.load(encoders_path)
    
    cat_cols = ['sex_age', 'jockey', 'trainer', 'weather', 'track', 'sire', 'bms', 'lineage', 'running_style']
    for c in cat_cols:
        if c in df_race.columns:
            le = encoders.get(c)
            df_race[c] = df_race[c].astype(str).fillna('unknown')
            if le:
                classes = list(le.classes_)
                df_race[c] = df_race[c].apply(lambda x: x if x in classes else 'unknown')
                df_race[c] = le.transform(df_race[c])
            else:
                df_race[c] = 0
                
    # モデルのロードと予測
    model_path = os.path.join(base_dir, 'src', 'models', 'lgbm_classifier.pkl')
    model = joblib.load(model_path)
    
    feature_cols = ['bracket', 'horse_num', 'sex_age', 'weight_constraint', 
                    'jockey', 'odds', 'popularity', 'horse_weight', 'trainer', 
                    'prev_rank', 'rest_days', 'weather', 'track', 'sire', 'bms', 'lineage',
                    'relative_last3f', 'running_style', 'is_makuri', 'is_long_trip', 'is_stay']
    
    X = df_race[feature_cols].copy()
    X.fillna(X.mean(numeric_only=True), inplace=True)
    X.fillna(0, inplace=True)
    
    # 3着以内に入る確率を予測
    probs = model.predict_proba(X)[:, 1]
    df_race['pred_prob'] = probs * 100 # %に変換
    
    # 確率順に並べ替え
    df_race = df_race.sort_values('pred_prob', ascending=False).reset_index(drop=True)
    
    # 予想順位と印の付与
    df_race['AI予想Rank'] = df_race.index + 1
    df_race['AI印'] = df_race['AI予想Rank'].apply(lambda r: get_mark(0, r))
    
    # 初期の文字列データ（エンコード前）が必要なため再度読み込む
    df_raw = pd.read_csv(shutuba_path)
    df_result = pd.merge(df_race[['horse_num', 'pred_prob', 'AI予想Rank', 'AI印']], df_raw, on='horse_num')
    df_result = df_result.sort_values('AI予想Rank')
    
    # 表示用Df
    display_df = pd.DataFrame({
        'AI印': df_result['AI印'],
        '馬番': df_result['horse_num'],
        '馬名': df_result['horse_name'],
        '複勝確率(%)': df_result['pred_prob'].round(1),
        'オッズ': df_result['odds'],
        '脚質': df_result['running_style'].replace({'front': '逃げ', 'stalker': '先行', 'closer': '差し', 'rear': '追込'}),
        '上がり偏差値': df_result['relative_last3f'],
        'AI予想Rank': df_result['AI予想Rank']
    })
    
    print(display_df.drop('AI予想Rank', axis=1).to_string(index=False))
    
    # Artifact生成
    generate_newspaper_artifact(display_df)

if __name__ == "__main__":
    predict_future_race()
