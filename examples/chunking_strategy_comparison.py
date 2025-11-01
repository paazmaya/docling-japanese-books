"""
Example usage of enhanced chunking strategies with all supported models.

This script demonstrates how to apply different chunking strategies to all models
in the docling-japanese-books project, including comparisons and recommendations.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from docling_japanese_books.config import config
from docling_japanese_books.enhanced_chunking import create_chunking_strategy

# Setup logging
logger = logging.getLogger(__name__)

# Type definitions for better type safety
ModelConfig = dict[str, str | None]
EvaluationResult = dict[str, Any]
ModelResults = dict[str, EvaluationResult]
ComparisonResults = dict[str, ModelResults]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_document() -> str:
    """Load a sample Japanese document for testing."""
    # Try to load from existing processed documents
    processed_dir = (
        Path(config.output.output_base_dir) / config.output.processed_output_dir
    )

    for file_path in processed_dir.glob("*.md"):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            if len(content) > 1000:  # Ensure sufficient content
                logger.info(f"Loaded sample document: {file_path.name}")
                return content[:5000]  # Limit to first 5000 chars for testing
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}")
            continue

    # Fallback sample text
    return """
    コンピュータサイエンスにおける自然言語処理（しぜんげんごしょり、Natural Language Processing、NLP）は、
    人間が日常的に使っている自然言語をコンピュータに処理させる一連の技術である。
    自然言語処理は言語学、コンピュータ科学、人工知能の学際的な分野である。

    近年、深層学習の発展により、自然言語処理技術は大幅に向上した。
    特に、Transformerアーキテクチャの登場は、機械翻訳、文書要約、質問応答などの
    タスクにおいて革命的な改善をもたらした。

    日本語の自然言語処理は、特有の課題を持つ。漢字、ひらがな、カタカナという
    三つの文字体系の混在、語順の柔軟性、敬語システムの複雑さなどが挙げられる。
    これらの特徴により、日本語専用の前処理技術や モデルの開発が重要である。

    埋め込み（エンベッディング）技術は、テキストを数値ベクトルに変換する技術であり、
    検索、分類、クラスタリングなどの下流タスクの基盤となっている。
    BGE-M3やJina Embeddings v4などの多言語対応モデルは、
    日本語テキストの処理において優れた性能を示している。
    """


def evaluate_chunking_strategy(
    document: str, model_name: str, strategy: str, task: str | None = None
) -> dict[str, Any]:  # type: ignore[misc]
    """Evaluate a specific chunking strategy."""
    logger.info(f"Evaluating {strategy} chunking with {model_name}")

    try:
        start_time = time.time()

        # Create chunking strategy
        chunker = create_chunking_strategy(model_name, strategy, task)

        # Process document
        chunks, embeddings = chunker.process_document(document, max_chunk_length=400)  # type: ignore[misc]

        processing_time = time.time() - start_time

        # Calculate metrics
        num_chunks = len(chunks)
        avg_chunk_length = sum(len(chunk) for chunk in chunks) / num_chunks
        embedding_dim = len(embeddings[0]) if embeddings else 0  # type: ignore[misc]

        # Context preservation score (similarity between consecutive chunks)
        context_scores = []
        for i in range(len(embeddings) - 1):  # type: ignore[misc]
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (  # type: ignore[misc]
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])  # type: ignore[misc]
            )
            context_scores.append(similarity)  # type: ignore[misc]

        avg_context_preservation = np.mean(context_scores) if context_scores else 0.0  # type: ignore[misc]

        return {
            "model_name": model_name,
            "strategy": strategy,
            "task": task,
            "success": True,
            "processing_time": processing_time,
            "num_chunks": num_chunks,
            "avg_chunk_length": avg_chunk_length,
            "embedding_dim": embedding_dim,
            "context_preservation": avg_context_preservation,
            "chunks_preview": chunks[:2],  # First 2 chunks for inspection
        }

    except Exception as e:
        logger.error(f"Failed to evaluate {strategy} with {model_name}: {e}")
        return {
            "model_name": model_name,
            "strategy": strategy,
            "task": task,
            "success": False,
            "error": str(e),
        }


def compare_all_strategies() -> ComparisonResults:
    """Compare all chunking strategies across all models."""
    logger.info("Starting comprehensive strategy comparison...")
    document = load_sample_document()

    # Model configurations with proper typing
    model_configs: list[ModelConfig] = [
        {"name": "BAAI/bge-m3", "task": None},
        {"name": "Snowflake/snowflake-arctic-embed-l-v2.0", "task": None},
        {"name": "jinaai/jina-embeddings-v4", "task": "retrieval"},
        {"name": "sentence-transformers/all-MiniLM-L6-v2", "task": None},
    ]

    # Chunking strategies to test
    strategies = ["late", "hybrid"]

    results: ComparisonResults = {}

    for model_config in model_configs:
        model_name: str = model_config["name"]  # type: ignore[assignment]
        task: str | None = model_config["task"]  # type: ignore[assignment]
        model_results: ModelResults = {}

        for strategy in strategies:
            result = evaluate_chunking_strategy(document, model_name, strategy, task)
            model_results[strategy] = result

        results[model_name] = model_results

    return results


def print_comparison_report(results: ComparisonResults) -> None:
    """Log a formatted comparison report."""
    _log_report_header()

    for model_name, model_results in results.items():
        _log_model_section(model_name, model_results)


def _log_report_header():
    """Log the report header."""
    logger.info("=" * 80)
    logger.info("CHUNKING STRATEGY COMPARISON REPORT")
    logger.info("=" * 80)


def _log_model_section(model_name: str, model_results: dict[str, Any]):  # type: ignore[misc]
    """Log results for a specific model."""
    logger.info(f"\n📊 MODEL: {model_name}")
    logger.info("-" * 60)

    for strategy, result in model_results.items():
        if result["success"]:
            _log_successful_strategy(strategy, result)
        else:
            _log_failed_strategy(strategy, result)


def _log_successful_strategy(strategy: str, result: dict[str, Any]):  # type: ignore[misc]
    """Log results for a successful strategy."""
    logger.info(f"\n✅ Strategy: {strategy.upper()}")
    if result.get("task"):
        logger.info(f"   Task: {result['task']}")
    logger.info(f"   Processing Time: {result['processing_time']:.2f}s")
    logger.info(f"   Chunks Generated: {result['num_chunks']}")
    logger.info(f"   Avg Chunk Length: {result['avg_chunk_length']:.0f} chars")
    logger.info(f"   Embedding Dimension: {result['embedding_dim']}")
    logger.info(f"   Context Preservation: {result['context_preservation']:.3f}")

    _log_chunk_preview(result)


def _log_failed_strategy(strategy: str, result: dict[str, Any]):  # type: ignore[misc]
    """Log results for a failed strategy."""
    logger.error(f"\n❌ Strategy: {strategy.upper()}")
    logger.error(f"   Error: {result['error']}")


def _log_chunk_preview(result: dict[str, Any]):  # type: ignore[misc]
    """Log a preview of the first chunk."""
    if result.get("chunks_preview"):
        first_chunk = result["chunks_preview"][0]
        preview = first_chunk[:100] + "..." if len(first_chunk) > 100 else first_chunk
        logger.debug(f"   First Chunk Preview: {preview}")


def generate_recommendations(results: ComparisonResults) -> list[str]:
    """Generate recommendations based on evaluation results."""
    recommendations: list[str] = []

    # Find best performing combinations
    successful_results: list[EvaluationResult] = []
    for _model_name, model_results in results.items():
        for _strategy, result in model_results.items():
            if result["success"]:
                successful_results.append(result)

    if not successful_results:
        recommendations.append(
            "⚠️  No successful evaluations found. Check model availability and dependencies."
        )
        return recommendations

    # Sort by context preservation score
    successful_results.sort(key=lambda x: x["context_preservation"], reverse=True)  # type: ignore[arg-type]

    best_result: EvaluationResult = successful_results[0]
    recommendations.append(
        f"🏆 Best Context Preservation: {best_result['model_name']} with {best_result['strategy']} chunking "
        f"(score: {best_result['context_preservation']:.3f})"
    )

    # Sort by processing speed
    successful_results.sort(key=lambda x: x["processing_time"])  # type: ignore[arg-type]
    fastest_result: EvaluationResult = successful_results[0]
    recommendations.append(
        f"⚡ Fastest Processing: {fastest_result['model_name']} with {fastest_result['strategy']} chunking "
        f"({fastest_result['processing_time']:.2f}s)"
    )

    # Model-specific recommendations
    jina_results: list[EvaluationResult] = [
        r for r in successful_results if "jina" in str(r["model_name"]).lower()
    ]  # type: ignore[operator]
    if jina_results:
        recommendations.append(
            "🎯 Jina v4 with quantization-aware training shows good balance of performance and efficiency"
        )

    bge_results: list[EvaluationResult] = [
        r for r in successful_results if "bge" in str(r["model_name"]).lower()
    ]  # type: ignore[operator]
    if bge_results:
        recommendations.append(
            "🌏 BGE-M3 is specifically optimized for multilingual content including Japanese"
        )

    # Late chunking recommendations
    late_chunking_results: list[EvaluationResult] = [
        r for r in successful_results if r["strategy"] == "late"
    ]
    if late_chunking_results:
        avg_context = np.mean(
            [r["context_preservation"] for r in late_chunking_results]  # type: ignore[misc]
        )
        recommendations.append(
            f"🔗 Late chunking shows average context preservation of {avg_context:.3f} across models"
        )

    return recommendations


def main():
    """Main execution function."""
    logger.info("Starting comprehensive chunking strategy evaluation...")
    logger.info(f"Using configuration from: {config}")

    # Run comparison
    results = compare_all_strategies()

    # Print report (using print for user-facing output)
    print_comparison_report(results)

    # Generate recommendations
    recommendations = generate_recommendations(results)

    logger.info("=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    for rec in recommendations:
        logger.info(f"{rec}")

    logger.info("=" * 80)
    logger.info("IMPLEMENTATION NOTES")
    logger.info("=" * 80)
    logger.info("1. Late chunking preserves better context but may be slower")
    logger.info("2. Hybrid strategies balance performance and quality")
    logger.info("3. Task-specific models (Jina v4) can be optimized for retrieval")
    logger.info("4. Consider your use case: speed vs. quality trade-offs")
    logger.info("5. Japanese text benefits from language-aware chunking")


if __name__ == "__main__":
    main()
