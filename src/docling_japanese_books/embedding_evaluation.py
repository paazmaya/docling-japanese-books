"""
Embedding evaluation framework for Japanese document processing.

This module provides tools to benchmark embedding models and chunking strategies,
particularly comparing traditional chunking with Late Chunking for Japanese text.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import config
from .late_chunking import LateChunkingProcessor


@dataclass
class EvaluationMetrics:
    """Metrics for embedding evaluation."""

    model_name: str
    chunking_method: str
    avg_cosine_similarity: float = 0.0
    std_cosine_similarity: float = 0.0
    processing_time: float = 0.0
    num_chunks: int = 0
    avg_chunk_length: int = 0
    context_preservation_score: float = 0.0
    japanese_specific_score: float = 0.0


@dataclass
class EvaluationResults:
    """Results from embedding evaluation."""

    document_id: str
    traditional_metrics: EvaluationMetrics
    late_chunking_metrics: EvaluationMetrics
    improvement_percentage: float = 0.0
    details: dict = field(default_factory=dict)


class EmbeddingEvaluator:
    """Evaluate and compare embedding approaches for Japanese documents."""

    def __init__(self):
        """Initialize the embedding evaluator."""
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Test queries for Japanese content evaluation
        self.japanese_test_queries = [
            "新しい機能は何ですか？",  # What are the new features?
            "システムの改善点について教えてください。",  # Tell me about system improvements.
            "この文書の主な内容は？",  # What is the main content of this document?
            "技術的な詳細を説明してください。",  # Please explain technical details.
            "問題の解決方法は？",  # What is the solution to the problem?
            "パフォーマンスの向上",  # Performance improvement
            "安定性とバグ修正",  # Stability and bug fixes
            "ユーザーエクスペリエンス",  # User experience
        ]

        # Initialize processors
        self.late_chunking = None
        self.traditional_model = None

    def load_models(self):
        """Load embedding models for comparison."""
        if self.late_chunking is None:
            self.logger.info("Loading BGE-M3 with Late Chunking...")
            self.late_chunking = LateChunkingProcessor()

        if self.traditional_model is None:
            self.logger.info("Loading traditional sentence-transformers model...")
            # Use the old model for comparison
            cache_folder = (
                Path(self.config.docling.artifacts_path).resolve()
                / "embeddings_comparison"
            )
            self.traditional_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(cache_folder)
            )

    def simple_traditional_chunking(
        self, document: str, max_length: int = 500
    ) -> list[str]:
        """Simple traditional chunking for comparison."""
        import re

        # Split by Japanese sentence endings
        sentences = re.split(r"[。！？]+", document)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + "。"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def evaluate_context_preservation(
        self, document: str, chunks: list[str], embeddings: list[np.ndarray]
    ) -> float:
        """Evaluate how well chunks preserve document context."""
        if len(chunks) < 2:
            return 1.0  # Perfect if only one chunk

        # Calculate similarity between consecutive chunks
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self.calculate_cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Higher similarity between consecutive chunks suggests better context preservation
        return np.mean(similarities) if similarities else 0.0

    def evaluate_japanese_specificity(
        self,
        chunks: list[str],
        embeddings: list[np.ndarray],
        use_bge_model: bool = False,
    ) -> float:
        """Evaluate performance on Japanese-specific queries."""
        similarities = []

        # Load models if not loaded
        self.load_models()

        for query in self.japanese_test_queries:
            # Get query embedding using the same model as chunks to ensure compatibility
            if use_bge_model:
                # Use BGE-M3 for query (matches Late Chunking embeddings)
                self.late_chunking.load_model()
                query_emb = self.late_chunking.model.encode(
                    [query], return_dense=True, return_sparse=False
                )["dense_vecs"][0]
            else:
                # Use sentence transformers (matches traditional embeddings)
                query_emb = self.traditional_model.encode(query)

            # Find best matching chunk
            chunk_similarities = []
            for emb in embeddings:
                sim = self.calculate_cosine_similarity(query_emb, emb)
                chunk_similarities.append(sim)

            if chunk_similarities:
                similarities.append(max(chunk_similarities))

        return np.mean(similarities) if similarities else 0.0

    def evaluate_document(self, document: str, doc_id: str) -> EvaluationResults:
        """Evaluate a document using both traditional and Late Chunking approaches."""
        self.load_models()
        self.logger.info(f"Evaluating document: {doc_id}")

        # Process with both approaches
        traditional_data = self._evaluate_traditional_approach(document)
        late_chunking_data = self._evaluate_late_chunking_approach(document)

        # Calculate metrics
        traditional_metrics = self._calculate_metrics(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunking_method="traditional",
            document=document,
            **traditional_data,
        )

        late_chunking_metrics = self._calculate_metrics(
            model_name="BAAI/bge-m3",
            chunking_method="late_chunking",
            document=document,
            **late_chunking_data,
        )

        # Calculate improvement
        improvement = self._calculate_improvement(
            traditional_metrics, late_chunking_metrics
        )

        return EvaluationResults(
            document_id=doc_id,
            traditional_metrics=traditional_metrics,
            late_chunking_metrics=late_chunking_metrics,
            improvement_percentage=improvement,
            details={
                "traditional_chunks": len(traditional_data["chunks"]),
                "late_chunking_chunks": len(late_chunking_data["chunks"]),
                "document_length": len(document),
                "queries_tested": len(self.japanese_test_queries),
            },
        )

    def _evaluate_traditional_approach(self, document: str) -> dict:
        """Evaluate traditional chunking approach."""
        start_time = time.time()
        chunks = self.simple_traditional_chunking(document)
        embeddings = [self.traditional_model.encode(chunk) for chunk in chunks]
        processing_time = time.time() - start_time

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "processing_time": processing_time,
        }

    def _evaluate_late_chunking_approach(self, document: str) -> dict:
        """Evaluate Late Chunking approach."""
        start_time = time.time()
        chunks, embeddings = self.late_chunking.process_document(document)
        processing_time = time.time() - start_time

        # Convert to numpy arrays
        embeddings_np = [
            emb if isinstance(emb, np.ndarray) else np.array(emb) for emb in embeddings
        ]

        return {
            "chunks": chunks,
            "embeddings": embeddings_np,
            "processing_time": processing_time,
        }

    def _calculate_metrics(
        self,
        model_name: str,
        chunking_method: str,
        document: str,
        chunks: list[str],
        embeddings: list[np.ndarray],
        processing_time: float,
    ) -> EvaluationMetrics:
        """Calculate evaluation metrics for an approach."""
        # Inter-chunk similarities
        similarities = []
        for i, emb1 in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings):
                if i != j:
                    sim = self.calculate_cosine_similarity(emb1, emb2)
                    similarities.append(sim)

        return EvaluationMetrics(
            model_name=model_name,
            chunking_method=chunking_method,
            avg_cosine_similarity=np.mean(similarities) if similarities else 0.0,
            std_cosine_similarity=np.std(similarities) if similarities else 0.0,
            processing_time=processing_time,
            num_chunks=len(chunks),
            avg_chunk_length=int(np.mean([len(c) for c in chunks])) if chunks else 0,
            context_preservation_score=self.evaluate_context_preservation(
                document, chunks, embeddings
            ),
            japanese_specific_score=self.evaluate_japanese_specificity(
                chunks, embeddings, use_bge_model=(chunking_method == "late_chunking")
            ),
        )

    def _calculate_improvement(
        self, traditional: EvaluationMetrics, late_chunking: EvaluationMetrics
    ) -> float:
        """Calculate improvement percentage."""
        if traditional.japanese_specific_score <= 0:
            return 0.0

        return (
            (
                late_chunking.japanese_specific_score
                - traditional.japanese_specific_score
            )
            / traditional.japanese_specific_score
            * 100
        )

    def run_comparison_study(
        self, documents: dict[str, str], output_path: Optional[Path] = None
    ) -> list[EvaluationResults]:
        """Run a comprehensive comparison study between embedding approaches."""
        results = []

        self.logger.info(f"Starting evaluation study with {len(documents)} documents")

        for doc_id, document in documents.items():
            try:
                result = self.evaluate_document(document, doc_id)
                results.append(result)

                self.logger.info(
                    f"Document {doc_id}: "
                    f"Japanese score improvement: {result.improvement_percentage:.1f}%, "
                    f"Context preservation: "
                    f"Traditional {result.traditional_metrics.context_preservation_score:.3f} -> "
                    f"Late Chunking {result.late_chunking_metrics.context_preservation_score:.3f}"
                )

            except Exception as e:
                self.logger.error(f"Failed to evaluate document {doc_id}: {e}")
                continue

        # Save results if output path provided
        if output_path:
            self.save_results(results, output_path)

        # Print summary
        self.print_summary(results)

        return results

    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return {
                k: self._convert_to_serializable(v) for k, v in obj.__dict__.items()
            }
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, results: list[EvaluationResults], output_path: Path):
        """Save evaluation results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_data = []
        for result in results:
            result_dict = {
                "document_id": result.document_id,
                "traditional_metrics": result.traditional_metrics.__dict__,
                "late_chunking_metrics": result.late_chunking_metrics.__dict__,
                "improvement_percentage": result.improvement_percentage,
                "details": result.details,
            }
            # Convert to JSON serializable format
            results_data.append(self._convert_to_serializable(result_dict))

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Evaluation results saved to: {output_path}")

    def print_summary(self, results: list[EvaluationResults]):
        """Print summary of evaluation results."""
        if not results:
            self.logger.warning("No results to summarize")
            return

        improvements = [r.improvement_percentage for r in results]
        context_improvements = [
            r.late_chunking_metrics.context_preservation_score
            - r.traditional_metrics.context_preservation_score
            for r in results
        ]
        japanese_scores_traditional = [
            r.traditional_metrics.japanese_specific_score for r in results
        ]
        japanese_scores_late = [
            r.late_chunking_metrics.japanese_specific_score for r in results
        ]

        print("\n" + "=" * 60)
        print("EMBEDDING EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Documents evaluated: {len(results)}")
        print(f"Average improvement in Japanese queries: {np.mean(improvements):.1f}%")
        print(
            f"Average context preservation improvement: {np.mean(context_improvements):.3f}"
        )
        print(
            f"Traditional Japanese score: {np.mean(japanese_scores_traditional):.3f} ± {np.std(japanese_scores_traditional):.3f}"
        )
        print(
            f"Late Chunking Japanese score: {np.mean(japanese_scores_late):.3f} ± {np.std(japanese_scores_late):.3f}"
        )

        best_improvement = max(results, key=lambda x: x.improvement_percentage)
        print(
            f"\nBest improvement: {best_improvement.document_id} (+{best_improvement.improvement_percentage:.1f}%)"
        )

        print("\nRecommendation:")
        if np.mean(improvements) > 5:
            print(
                "✅ Late Chunking shows significant improvement - RECOMMENDED for production"
            )
        elif np.mean(improvements) > 0:
            print(
                "⚠️  Late Chunking shows modest improvement - consider for Japanese-heavy workflows"
            )
        else:
            print(
                "❌ Late Chunking does not show improvement - stick with traditional approach"
            )

        print("=" * 60)


def create_sample_japanese_documents() -> dict[str, str]:
    """Create sample Japanese documents for testing."""
    return {
        "milvus_release": """
        Milvus 2.4.13では動的レプリカ負荷機能を導入しており、ユーザーはコレクションのリリースと
        リロードを行うことなく、コレクションレプリカの数を調整できるようになりました。このバージョンでは、
        バルクインポート、式の解析、負荷分散、障害復旧に関連するいくつかの重要なバグも修正されています。
        さらに、MMAPリソース使用量とインポートパフォーマンスが大幅に改善され、システム全体の効率が
        向上しています。より良いパフォーマンスと安定性のため、このリリースへのアップグレードを強く推奨します。
        """,
        "technical_manual": """
        システムアーキテクチャについて説明します。本システムは分散型ベクトルデータベースとして設計されており、
        高いスケーラビリティと可用性を提供します。主要なコンポーネントには、クエリノード、データノード、
        インデックスノード、ルートコーディネータが含まれます。各ノードは独立してスケールでき、
        システム全体の負荷に応じて動的に調整されます。データの永続化にはオブジェクトストレージが使用され、
        メタデータ管理にはetcdが採用されています。
        """,
        "user_guide": """
        このガイドでは、AIアプリケーション開発におけるベクトル検索の基本的な使用方法について
        説明します。まず、データの前処理を行い、適切な埋め込みモデルを選択します。次に、
        ベクトル化されたデータをコレクションに挿入し、効率的な検索のためのインデックスを構築します。
        検索時は、クエリベクトルと最も類似したベクトルを見つけるために、コサイン類似度や
        ユークリッド距離などの距離メトリクスを使用します。パフォーマンスの最適化には、
        適切なインデックスタイプの選択とパラメータのチューニングが重要です。
        """,
    }


# CLI command for running evaluations
def main():
    """Main function for running embedding evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate embedding approaches for Japanese documents"
    )
    parser.add_argument("--output", "-o", type=Path, help="Output file for results")
    parser.add_argument(
        "--documents", "-d", type=Path, help="JSON file with documents to evaluate"
    )

    args = parser.parse_args()

    evaluator = EmbeddingEvaluator()

    # Load documents
    if args.documents and args.documents.exists():
        with args.documents.open("r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        print("Using sample Japanese documents for evaluation...")
        documents = create_sample_japanese_documents()

    # Run evaluation
    results = evaluator.run_comparison_study(
        documents, output_path=args.output or Path("embedding_evaluation_results.json")
    )

    return results


if __name__ == "__main__":
    main()
