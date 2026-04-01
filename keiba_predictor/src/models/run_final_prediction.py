"""
三冠馬 最終予測実行 - 2026年4月4日・5日 実際の出走馬データ使用
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from triple_crown import TripleCrown
from race_entries_real_apr2026 import ALL_RACES

def main():
    engine = TripleCrown()

    print("=" * 66)
    print("  🏇 2026年4月4日・5日 最終予想（実際の出走馬使用）")
    print("  ■ 使用モデル: 三冠馬 v2.0 (2023-2026/Q1 全データ学習済)")
    print("  ■ SHAP分析: 個体別選出根拠を出力")
    print("=" * 66)

    for race_id, race_meta in ALL_RACES.items():
        report = engine.format_prediction(race_id, race_meta)
        print(report)


if __name__ == '__main__':
    main()
