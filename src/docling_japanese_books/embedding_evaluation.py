"""
Embedding evaluation framework for Japanese document processing.

This module provides tools to benchmark embedding models and chunking strategies,
particularly comparing BGE-M3 with Late Chunking against Snowflake Arctic Embed
and traditional models using real Japanese documents.

Supports evaluation of:
- BGE-M3 with Late Chunking (https://huggingface.co/BAAI/bge-m3)
- Snowflake Arctic Embed L v2.0 (https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
- Traditional all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The evaluation uses real Japanese documents processed through Docling for authentic
performance measurement on historical martial arts texts and technical content.
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

# Jina v4 task constants
# Supported tasks: 'retrieval', 'text-matching', 'code'
JINA_TASK_RETRIEVAL = "retrieval"


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
    snowflake_arctic_metrics: EvaluationMetrics
    jina_v4_metrics: EvaluationMetrics
    bge_m3_improvement: float = 0.0
    snowflake_improvement: float = 0.0
    jina_v4_improvement: float = 0.0
    best_model: str = ""
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
        self.snowflake_arctic_model = None
        self.jina_v4_model = None

    def load_models(self):
        """Load embedding models for comparison."""
        if self.late_chunking is None:
            self.logger.info("Loading BGE-M3 with Late Chunking...")
            self.late_chunking = LateChunkingProcessor()

        if self.traditional_model is None:
            self.logger.info("Loading traditional sentence-transformers model...")
            # Use the old model for comparison
            # Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
            cache_folder = (
                Path(self.config.docling.artifacts_path).resolve()
                / "embeddings_comparison"
            )
            self.traditional_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(cache_folder)
            )

        if self.snowflake_arctic_model is None:
            self.logger.info("Loading Snowflake Arctic Embed L v2.0...")
            # Model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0
            cache_folder = (
                Path(self.config.docling.artifacts_path).resolve()
                / "embeddings_comparison"
            )
            self.snowflake_arctic_model = SentenceTransformer(
                "Snowflake/snowflake-arctic-embed-l-v2.0",
                cache_folder=str(cache_folder),
            )

        if self.jina_v4_model is None:
            self.logger.info("Loading Jina Embeddings v4...")
            # Model: https://huggingface.co/jinaai/jina-embeddings-v4
            # Features quantization-aware training for improved efficiency
            # Supported tasks: 'retrieval', 'text-matching', 'code'
            cache_folder = (
                Path(self.config.docling.artifacts_path).resolve()
                / "embeddings_comparison"
            )
            self.jina_v4_model = SentenceTransformer(
                "jinaai/jina-embeddings-v4",
                cache_folder=str(cache_folder),
                trust_remote_code=True,
                model_kwargs={"default_task": JINA_TASK_RETRIEVAL},
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
        model_type: str = "traditional",
    ) -> float:
        """Evaluate performance on Japanese-specific queries."""
        similarities = []

        # Load models if not loaded
        self.load_models()

        for query in self.japanese_test_queries:
            # Get query embedding using the same model as chunks to ensure compatibility
            if model_type == "bge_m3":
                # Use BGE-M3 for query (matches Late Chunking embeddings)
                self.late_chunking.load_model()
                query_emb = self.late_chunking.model.encode(
                    [query], return_dense=True, return_sparse=False
                )["dense_vecs"][0]
            elif model_type == "snowflake_arctic":
                # Use Snowflake Arctic for query
                query_emb = self.snowflake_arctic_model.encode(query)
            elif model_type == "jina_v4":
                # Use Jina Embeddings v4 for query with specific task
                query_emb = self.jina_v4_model.encode(query, task=JINA_TASK_RETRIEVAL)
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

    def _get_model_type(self, chunking_method: str) -> str:
        """Get model type based on chunking method."""
        if chunking_method == "late_chunking":
            return "bge_m3"
        elif chunking_method == "snowflake_arctic":
            return "snowflake_arctic"
        elif chunking_method == "jina_v4":
            return "jina_v4"
        else:
            return "traditional"

    def evaluate_document(self, document: str, doc_id: str) -> EvaluationResults:
        """Evaluate a document using traditional, Late Chunking, Snowflake Arctic, and Jina v4 approaches."""
        self.load_models()
        self.logger.info(f"Evaluating document: {doc_id}")

        # Process with all four approaches
        traditional_data = self._evaluate_traditional_approach(document)
        late_chunking_data = self._evaluate_late_chunking_approach(document)
        snowflake_data = self._evaluate_snowflake_arctic_approach(document)
        jina_v4_data = self._evaluate_jina_v4_approach(document)

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

        snowflake_arctic_metrics = self._calculate_metrics(
            model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
            chunking_method="snowflake_arctic",
            document=document,
            **snowflake_data,
        )

        jina_v4_metrics = self._calculate_metrics(
            model_name="jinaai/jina-embeddings-v4",
            chunking_method="jina_v4",
            document=document,
            **jina_v4_data,
        )

        # Calculate improvements
        bge_improvement = self._calculate_improvement(
            traditional_metrics, late_chunking_metrics
        )
        snowflake_improvement = self._calculate_improvement(
            traditional_metrics, snowflake_arctic_metrics
        )
        jina_v4_improvement = self._calculate_improvement(
            traditional_metrics, jina_v4_metrics
        )

        # Determine best model
        best_model = self._determine_best_model(
            traditional_metrics,
            late_chunking_metrics,
            snowflake_arctic_metrics,
            jina_v4_metrics,
        )

        return EvaluationResults(
            document_id=doc_id,
            traditional_metrics=traditional_metrics,
            late_chunking_metrics=late_chunking_metrics,
            snowflake_arctic_metrics=snowflake_arctic_metrics,
            jina_v4_metrics=jina_v4_metrics,
            bge_m3_improvement=bge_improvement,
            snowflake_improvement=snowflake_improvement,
            jina_v4_improvement=jina_v4_improvement,
            best_model=best_model,
            details={
                "traditional_chunks": len(traditional_data["chunks"]),
                "late_chunking_chunks": len(late_chunking_data["chunks"]),
                "snowflake_chunks": len(snowflake_data["chunks"]),
                "jina_v4_chunks": len(jina_v4_data["chunks"]),
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

    def _evaluate_snowflake_arctic_approach(self, document: str) -> dict:
        """Evaluate Snowflake Arctic approach with traditional chunking."""
        start_time = time.time()
        chunks = self.simple_traditional_chunking(document)
        embeddings = [self.snowflake_arctic_model.encode(chunk) for chunk in chunks]
        processing_time = time.time() - start_time

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "processing_time": processing_time,
        }

    def _evaluate_jina_v4_approach(self, document: str) -> dict:
        """Evaluate Jina Embeddings v4 approach with traditional chunking and proper task specification."""
        start_time = time.time()
        chunks = self.simple_traditional_chunking(document)
        # Use proper task specification for document passages
        embeddings = [
            self.jina_v4_model.encode(chunk, task=JINA_TASK_RETRIEVAL)
            for chunk in chunks
        ]
        processing_time = time.time() - start_time

        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "processing_time": processing_time,
        }

    def _evaluate_jina_v4_late_chunking_approach(self, document: str) -> dict:
        """Evaluate Jina v4 with native late chunking if supported."""
        start_time = time.time()

        # Try to use Jina's native late chunking approach
        try:
            # Check if the model supports late chunking parameters
            # This is experimental - Jina v4 may support this natively
            # Note: late_chunking parameter may not be available in sentence-transformers interface
            # This would require direct API calls to Jina's service

            # For now, fall back to our own chunking approach but with proper tasks
            chunks = self.simple_traditional_chunking(document)
            embeddings = [
                self.jina_v4_model.encode(chunk, task=JINA_TASK_RETRIEVAL)
                for chunk in chunks
            ]

        except Exception:
            # Fall back to standard chunking if late chunking is not available
            chunks = self.simple_traditional_chunking(document)
            embeddings = [
                self.jina_v4_model.encode(chunk, task=JINA_TASK_RETRIEVAL)
                for chunk in chunks
            ]

        processing_time = time.time() - start_time

        return {
            "chunks": chunks,
            "embeddings": embeddings,
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
                chunks, embeddings, model_type=self._get_model_type(chunking_method)
            ),
        )

    def _calculate_improvement(
        self, baseline: EvaluationMetrics, comparison: EvaluationMetrics
    ) -> float:
        """Calculate improvement percentage."""
        if baseline.japanese_specific_score <= 0:
            return 0.0

        return (
            (comparison.japanese_specific_score - baseline.japanese_specific_score)
            / baseline.japanese_specific_score
            * 100
        )

    def _determine_best_model(
        self,
        traditional: EvaluationMetrics,
        bge_m3: EvaluationMetrics,
        snowflake: EvaluationMetrics,
        jina_v4: EvaluationMetrics,
    ) -> str:
        """Determine which model performed best based on Japanese-specific score."""
        scores = [
            (traditional.japanese_specific_score, "Traditional (all-MiniLM-L6-v2)"),
            (bge_m3.japanese_specific_score, "BGE-M3 (Late Chunking)"),
            (snowflake.japanese_specific_score, "Snowflake Arctic Embed L v2.0"),
            (jina_v4.japanese_specific_score, "Jina Embeddings v4"),
        ]

        # Sort by score (descending) and return the best model name
        _, best_model = max(scores, key=lambda x: x[0])
        return best_model

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
                    f"BGE-M3 improvement: {result.bge_m3_improvement:.1f}%, "
                    f"Snowflake improvement: {result.snowflake_improvement:.1f}%, "
                    f"Jina v4 improvement: {result.jina_v4_improvement:.1f}%, "
                    f"Best model: {result.best_model}"
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
                "snowflake_arctic_metrics": result.snowflake_arctic_metrics.__dict__,
                "jina_v4_metrics": result.jina_v4_metrics.__dict__,
                "bge_m3_improvement": result.bge_m3_improvement,
                "snowflake_improvement": result.snowflake_improvement,
                "jina_v4_improvement": result.jina_v4_improvement,
                "best_model": result.best_model,
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

        bge_improvements = [r.bge_m3_improvement for r in results]
        snowflake_improvements = [r.snowflake_improvement for r in results]
        jina_v4_improvements = [r.jina_v4_improvement for r in results]

        japanese_scores_traditional = [
            r.traditional_metrics.japanese_specific_score for r in results
        ]
        japanese_scores_bge = [
            r.late_chunking_metrics.japanese_specific_score for r in results
        ]
        japanese_scores_snowflake = [
            r.snowflake_arctic_metrics.japanese_specific_score for r in results
        ]
        japanese_scores_jina_v4 = [
            r.jina_v4_metrics.japanese_specific_score for r in results
        ]

        # Count best models
        model_wins = {}
        for result in results:
            model = result.best_model
            model_wins[model] = model_wins.get(model, 0) + 1

        print("\n" + "=" * 80)
        print("COMPREHENSIVE EMBEDDING EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Documents evaluated: {len(results)}")
        print()
        print("📊 JAPANESE-SPECIFIC QUERY PERFORMANCE:")
        print(
            f"Traditional (all-MiniLM-L6-v2): {np.mean(japanese_scores_traditional):.3f} ± {np.std(japanese_scores_traditional):.3f}"
        )
        print(
            f"BGE-M3 (Late Chunking):        {np.mean(japanese_scores_bge):.3f} ± {np.std(japanese_scores_bge):.3f}"
        )
        print(
            f"Snowflake Arctic Embed L v2.0: {np.mean(japanese_scores_snowflake):.3f} ± {np.std(japanese_scores_snowflake):.3f}"
        )
        print(
            f"Jina Embeddings v4:            {np.mean(japanese_scores_jina_v4):.3f} ± {np.std(japanese_scores_jina_v4):.3f}"
        )
        print()
        print("📈 IMPROVEMENT OVER TRADITIONAL:")
        print(
            f"BGE-M3 improvement:      {np.mean(bge_improvements):.1f}% ± {np.std(bge_improvements):.1f}%"
        )
        print(
            f"Snowflake improvement:   {np.mean(snowflake_improvements):.1f}% ± {np.std(snowflake_improvements):.1f}%"
        )
        print(
            f"Jina v4 improvement:     {np.mean(jina_v4_improvements):.1f}% ± {np.std(jina_v4_improvements):.1f}%"
        )
        print()
        print("🏆 MODEL WINS (best performance per document):")
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            print(
                f"{model}: {wins}/{len(results)} documents ({wins / len(results) * 100:.1f}%)"
            )

        # Best performing model overall
        best_bge = max(results, key=lambda x: x.bge_m3_improvement)
        best_snowflake = max(results, key=lambda x: x.snowflake_improvement)
        best_jina_v4 = max(results, key=lambda x: x.jina_v4_improvement)

        print()
        print("🚀 BEST INDIVIDUAL PERFORMANCES:")
        print(
            f"BGE-M3 best:      {best_bge.document_id} (+{best_bge.bge_m3_improvement:.1f}%)"
        )
        print(
            f"Snowflake best:   {best_snowflake.document_id} (+{best_snowflake.snowflake_improvement:.1f}%)"
        )
        print(
            f"Jina v4 best:     {best_jina_v4.document_id} (+{best_jina_v4.jina_v4_improvement:.1f}%)"
        )

        print()
        print("💡 RECOMMENDATIONS:")

        avg_bge = np.mean(bge_improvements)
        avg_snowflake = np.mean(snowflake_improvements)
        avg_jina_v4 = np.mean(jina_v4_improvements)

        # Find the best performing model
        models_scores = [
            (avg_bge, "BGE-M3 with Late Chunking", "BGE-M3"),
            (avg_snowflake, "Snowflake Arctic Embed L v2.0", "Snowflake Arctic"),
            (avg_jina_v4, "Jina Embeddings v4", "Jina v4"),
        ]
        best_score, best_name, best_short = max(models_scores, key=lambda x: x[0])

        if best_score > 5:
            print(f"✅ {best_name} shows BEST performance - RECOMMENDED")
        elif best_score > 0:
            print(
                f"⚠️  {best_short} shows modest improvement - consider for Japanese-heavy workflows"
            )
        else:
            print("❌ Advanced models do not show significant improvement")

        print()
        print("🔄 NEXT STEPS:")
        if best_short == "Jina v4":
            print(
                "• Consider implementing Jina Embeddings v4 as primary embedding model"
            )
            print("• Leverage quantization-aware training benefits for Japanese text")
            print("• Test on larger document collections to validate performance")
        elif best_short == "Snowflake Arctic":
            print(
                "• Consider implementing Snowflake Arctic Embed L v2.0 as primary embedding model"
            )
            print("• Test on larger document collections to validate performance")
            print(
                "• Update vector database schema for optimal Snowflake embedding dimensions"
            )
        else:
            print("• Current BGE-M3 implementation appears optimal")
            print(
                "• Consider hybrid approach: BGE-M3 for chunking, other models for specific use cases"
            )

        print("=" * 60)


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
        # Process PDF files using DocumentProcessor with vision model
        logger = logging.getLogger(__name__)
        test_docs_path = Path("test_docs")
        documents = {}

        if test_docs_path.exists():
            pdf_files = list(test_docs_path.glob("*.pdf"))
            if pdf_files:
                logger.info(
                    f"Processing {len(pdf_files)} PDF files using vision model..."
                )

                try:
                    # Import and initialize DocumentProcessor
                    from .processor import DocumentProcessor

                    # Process each PDF individually to extract text content
                    for pdf_file in pdf_files:
                        content = None

                        try:
                            logger.info(
                                f"Processing {pdf_file.name} with vision model..."
                            )

                            # Initialize processor for single file
                            processor = DocumentProcessor()

                            # Process the single PDF file
                            results = processor.process_files([pdf_file])

                            if results.success_count > 0:
                                # Try to get the processed content from the results
                                # The processor should have generated markdown content
                                # We need to extract it from the output files

                                output_dir = Path("output") / pdf_file.stem
                                markdown_file = output_dir / f"{pdf_file.stem}.md"

                                if markdown_file.exists():
                                    with markdown_file.open("r", encoding="utf-8") as f:
                                        content = f.read()
                                        if content.strip():
                                            documents[pdf_file.stem] = content
                                            logger.info(
                                                f"Successfully processed {pdf_file.name}: {len(content)} characters"
                                            )
                                        else:
                                            logger.warning(
                                                f"Empty content from {pdf_file.name}"
                                            )
                                else:
                                    logger.warning(
                                        f"No output markdown file found for {pdf_file.name}"
                                    )
                            else:
                                logger.warning(f"Failed to process {pdf_file.name}")

                        except Exception as e:
                            logger.warning(
                                f"Error processing {pdf_file.name} with vision model: {e}"
                            )

                        # If vision processing failed, try fallback text extraction
                        if not content:
                            raise ValueError(
                                f"Failed to extract content from {pdf_file.name}"
                            )

                except Exception as e:
                    logger.error(f"Failed to initialize DocumentProcessor: {e}")

            else:
                logger.warning("No PDF files found in test_docs/")
        else:
            logger.error("test_docs/ folder not found")

    # Run evaluation
    results = evaluator.run_comparison_study(
        documents, output_path=args.output or Path("embedding_evaluation_results.json")
    )

    return results


if __name__ == "__main__":
    main()
