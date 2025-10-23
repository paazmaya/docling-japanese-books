#!/usr/bin/env python3
"""
Evaluate Snowflake Arctic Embed L v2.0 against current BGE-M3 implementation.

This script runs a comprehensive evaluation comparing:
1. Traditional approach (all-MiniLM-L6-v2): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. BGE-M3 with Late Chunking (current implementation): https://huggingface.co/BAAI/bge-m3
3. Snowflake Arctic Embed L v2.0 with traditional chunking: https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0

Usage:
    python scripts/evaluate_snowflake_arctic.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("embedding_evaluation_snowflake.log"),
        ],
    )


def load_test_documents():
    """Load Japanese test documents for evaluation."""
    # Sample Japanese documents representing different styles and complexity levels
    documents = {
        "technical_manual": """
新しいソフトウェアバージョン2.5の技術仕様書

概要
このバージョンでは、パフォーマンスの大幅な改善と新機能の追加を行いました。
主要な変更点は以下の通りです。

機能改善
- 処理速度が前バージョンと比較して40%向上しました
- メモリ使用量を30%削減しました
- 新しいAPIエンドポイントを5つ追加しました
- データベース接続の安定性を向上させました

新機能
1. リアルタイム分析機能
   - ダッシュボードでのリアルタイムデータ表示
   - カスタマイズ可能なチャート機能
   - エクスポート機能（PDF、Excel対応）

2. セキュリティ強化
   - 二段階認証の実装
   - SSL/TLS暗号化の強化
   - アクセスログの詳細化

3. ユーザビリティ改善
   - インターフェースのレスポンシブデザイン対応
   - 多言語サポート（日本語、英語、中国語）
   - キーボードショートカットの追加

技術詳細
システム要件：
- CPU: Intel i5 2.4GHz以上またはAMD Ryzen 5以上
- メモリ: 8GB RAM以上推奨
- ストレージ: 500MB以上の空き容量
- OS: Windows 10/11、macOS 10.15以降、Ubuntu 18.04以降

インストール手順：
1. 既存バージョンのバックアップを作成してください
2. インストーラーをダウンロードし実行します
3. 設定ファイルの移行ウィザードに従ってください
4. 初回起動時にライセンス認証を行います

トラブルシューティング：
問題が発生した場合は、以下を確認してください：
- システム要件を満たしているか
- ファイアウォール設定が適切か  
- データベース接続設定が正しいか

サポート連絡先：support@example.com
        """,
        
        "business_report": """
2024年第3四半期業績報告書

エグゼクティブサマリー
当四半期における業績は予想を上回る結果となりました。
売上高は前年同期比15%増となり、利益率も2%向上しました。

主要指標
- 売上高: 125億円（前年同期比+15%）
- 営業利益: 18億円（前年同期比+22%）
- 純利益: 12億円（前年同期比+18%）
- 従業員数: 2,450名（前期末比+120名）

事業セグメント別分析

1. デジタルソリューション事業（売上の55%）
   売上高: 69億円（+25%）
   - クラウドサービス需要の急増
   - AI・機械学習ソリューションの好調
   - 新規顧客獲得数: 145社

2. コンサルティング事業（売上の30%）  
   売上高: 37.5億円（+8%)
   - 大手企業向けDX支援案件の増加
   - リピート率: 92%（前期比+3%）

3. 教育・研修事業（売上の15%）
   売上高: 18.5億円（+3%）
   - オンライン研修の需要拡大
   - 新コース開設: 12講座

市場環境と競合状況
デジタル変革の加速により市場は拡大傾向にあります。
主要競合企業：
- A社: 市場シェア28%（当社は18%で3位）
- B社: 市場シェア22%  
- C社: 市場シェア15%

今後の展望
第4四半期に向けては以下の施策を推進します：
1. 新製品3つのリリース準備
2. 海外市場への本格展開検討
3. パートナー企業との提携拡大
4. 人材採用の積極化（年間200名予定）

リスク要因
- 人材不足による成長制約
- 競合他社との価格競争激化
- 経済情勢の不安定化による企業投資意欲の減退

株主の皆様への感謝とともに、持続的成長に向けた取り組みを継続してまいります。
        """,
        
        "research_paper": """
日本語自然言語処理における埋め込みモデルの性能評価

要旨
本研究では、日本語テキストの意味理解において、異なる埋め込みモデルの性能を比較検討した。
特に、文書分割手法と埋め込み生成手法の組み合わせによる影響を詳細に分析した。

1. はじめに
自然言語処理（NLP）の分野では、テキストの数値表現である埋め込み（embedding）が重要な役割を果たしている。
日本語は語彙の多様性と文法の複雑性により、効果的な埋め込み生成に特有の課題がある。

本研究の目的は以下の通りである：
- 現行の埋め込みモデルの日本語テキストにおける性能評価
- 文書分割手法（chunking strategy）の影響分析  
- 実用的な推奨システムの構築

2. 関連研究
近年の transformer ベースモデルの発展により、多言語対応モデルが注目されている。
主要なモデルとして以下が挙げられる：

2.1 BERT系モデル
- 日本語特化型：tohoku-bert, rinna-bert
- 多言語型：multilingual-BERT, xlm-roberta

2.2 新世代埋め込みモデル  
- BGE（BAAI General Embedding）シリーズ
- E5（Microsoft）シリーズ
- Snowflake Arctic Embed シリーズ

3. 実験設計

3.1 データセット
評価用データセットとして以下を使用した：
- 技術文書：500件（平均1,200文字）
- ビジネス文書：300件（平均800文字）
- 学術論文：200件（平均2,000文字）

3.2 評価指標
- 意味的類似度（cosine similarity）
- 検索精度（retrieval accuracy）
- 処理時間（processing time）
- 文脈保持度（context preservation）

3.3 実験手法
各モデルに対して以下の手順で評価を実施：
1. 文書の前処理と分割
2. 埋め込みベクトルの生成
3. クエリ-文書間類似度の計算
4. 評価指標の算出と統計分析

4. 結果

4.1 全体的な性能比較
BGE-M3モデルが最も高い性能を示し、特に技術文書において優秀だった。
平均類似度スコア：
- BGE-M3: 0.847
- Snowflake Arctic: 0.831  
- multilingual-BERT: 0.792

4.2 文書種別による差異
技術文書では専門用語の理解において BGE-M3 が優位性を示した。
一方、ビジネス文書では Snowflake Arctic が競合する性能を発揮。

5. 考察
実験結果から以下の知見が得られた：

5.1 モデル選択の指針
- 技術文書中心：BGE-M3推奨
- 汎用的用途：Snowflake Arctic も検討価値あり
- 処理速度重視：従来のBERTベースモデルも選択肢

5.2 実装上の考慮事項
- 計算資源の要件
- 推論速度とバッチサイズの最適化
- メモリ使用量の管理

6. まとめ
日本語埋め込みモデルの性能は用途により最適解が異なることが明らかになった。
今後は、ドメイン特化型の fine-tuning や hybrid approach の検討が重要である。

謝辞
本研究の実施にあたり、多大なご協力をいただいた関係各位に深謝いたします。

参考文献
[1] Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers"  
[2] Reimers, N. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
[3] 黒橋禎夫 "日本語自然言語処理の現状と課題"
        """
    }
    
    return documents


def main():
    """Run Snowflake Arctic embedding evaluation."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Snowflake Arctic Embed L v2.0 evaluation")
    
    # Load test documents
    documents = load_test_documents()
    logger.info(f"Loaded {len(documents)} test documents")
    
    # Initialize evaluator
    evaluator = EmbeddingEvaluator()
    
    # Run evaluation
    results_path = Path("embedding_evaluation_snowflake_results.json")
    results = evaluator.run_comparison_study(documents, results_path)
    
    logger.info(f"Evaluation completed. Results saved to {results_path}")
    
    # Additional analysis
    print("\n🎯 DETAILED MODEL COMPARISON:")
    for result in results:
        print(f"\n📄 Document: {result.document_id}")
        print(f"   Traditional: {result.traditional_metrics.japanese_specific_score:.3f}")
        print(f"   BGE-M3:      {result.late_chunking_metrics.japanese_specific_score:.3f} ({result.bge_m3_improvement:+.1f}%)")
        print(f"   Snowflake:   {result.snowflake_arctic_metrics.japanese_specific_score:.3f} ({result.snowflake_improvement:+.1f}%)")
        print(f"   Winner:      {result.best_model}")


if __name__ == "__main__":
    main()