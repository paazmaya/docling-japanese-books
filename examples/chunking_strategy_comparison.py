"""
Example usage of enhanced chunking strategies with all supported models.

This script demonstrates how to apply different chunking strategies to all models
in the docling-japanese-books project, including comparisons and recommendations.
"""

import logging
import time
from pathlib import Path

import numpy as np

from docling_japanese_books.config import config
from docling_japanese_books.enhanced_chunking import create_chunking_strategy

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
    document: str, model_name: str, strategy: str, task: str = None
) -> dict:
    """Evaluate a specific chunking strategy."""
    logger.info(f"Evaluating {strategy} chunking with {model_name}")

    try:
        start_time = time.time()

        # Create chunking strategy
        chunker = create_chunking_strategy(model_name, strategy, task)

        # Process document
        chunks, embeddings = chunker.process_document(document, max_chunk_length=400)

        processing_time = time.time() - start_time

        # Calculate metrics
        num_chunks = len(chunks)
        avg_chunk_length = sum(len(chunk) for chunk in chunks) / num_chunks
        embedding_dim = len(embeddings[0]) if embeddings else 0

        # Context preservation score (similarity between consecutive chunks)
        context_scores = []
        for i in range(len(embeddings) - 1):
            similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            context_scores.append(similarity)

        avg_context_preservation = np.mean(context_scores) if context_scores else 0.0

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


def compare_all_strategies() -> dict:
    """Compare all chunking strategies across all models."""
    document = load_sample_document()

    # Model configurations
    model_configs = [
        {"name": "BAAI/bge-m3", "task": None},
        {"name": "Snowflake/snowflake-arctic-embed-l-v2.0", "task": None},
        {"name": "jinaai/jina-embeddings-v4", "task": "retrieval"},
        {"name": "sentence-transformers/all-MiniLM-L6-v2", "task": None},
    ]

    # Chunking strategies to test
    strategies = ["late", "hybrid"]

    results = {}

    for model_config in model_configs:
        model_name = model_config["name"]
        task = model_config["task"]
        model_results = {}

        for strategy in strategies:
            result = evaluate_chunking_strategy(document, model_name, strategy, task)
            model_results[strategy] = result

        results[model_name] = model_results

    return results


def print_comparison_report(results: dict):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("CHUNKING STRATEGY COMPARISON REPORT")
    print("=" * 80)

    for model_name, model_results in results.items():
        print(f"\n📊 MODEL: {model_name}")
        print("-" * 60)

        for strategy, result in model_results.items():
            if result["success"]:
                print(f"\n✅ Strategy: {strategy.upper()}")
                if result.get("task"):
                    print(f"   Task: {result['task']}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                print(f"   Chunks Generated: {result['num_chunks']}")
                print(f"   Avg Chunk Length: {result['avg_chunk_length']:.0f} chars")
                print(f"   Embedding Dimension: {result['embedding_dim']}")
                print(f"   Context Preservation: {result['context_preservation']:.3f}")

                # Show first chunk as example
                if result.get("chunks_preview"):
                    preview = (
                        result["chunks_preview"][0][:100] + "..."
                        if len(result["chunks_preview"][0]) > 100
                        else result["chunks_preview"][0]
                    )
                    print(f"   First Chunk Preview: {preview}")
            else:
                print(f"\n❌ Strategy: {strategy.upper()}")
                print(f"   Error: {result['error']}")


def generate_recommendations(results: dict) -> list[str]:
    """Generate recommendations based on evaluation results."""
    recommendations = []

    # Find best performing combinations
    successful_results = []
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
    successful_results.sort(key=lambda x: x["context_preservation"], reverse=True)

    best_result = successful_results[0]
    recommendations.append(
        f"🏆 Best Context Preservation: {best_result['model_name']} with {best_result['strategy']} chunking "
        f"(score: {best_result['context_preservation']:.3f})"
    )

    # Sort by processing speed
    successful_results.sort(key=lambda x: x["processing_time"])
    fastest_result = successful_results[0]
    recommendations.append(
        f"⚡ Fastest Processing: {fastest_result['model_name']} with {fastest_result['strategy']} chunking "
        f"({fastest_result['processing_time']:.2f}s)"
    )

    # Model-specific recommendations
    jina_results = [r for r in successful_results if "jina" in r["model_name"].lower()]
    if jina_results:
        recommendations.append(
            "🎯 Jina v4 with quantization-aware training shows good balance of performance and efficiency"
        )

    bge_results = [r for r in successful_results if "bge" in r["model_name"].lower()]
    if bge_results:
        recommendations.append(
            "🌏 BGE-M3 is specifically optimized for multilingual content including Japanese"
        )

    # Late chunking recommendations
    late_chunking_results = [r for r in successful_results if r["strategy"] == "late"]
    if late_chunking_results:
        avg_context = np.mean(
            [r["context_preservation"] for r in late_chunking_results]
        )
        recommendations.append(
            f"🔗 Late chunking shows average context preservation of {avg_context:.3f} across models"
        )

    return recommendations


def main():
    """Main execution function."""
    print("Starting comprehensive chunking strategy evaluation...")
    print(f"Using configuration from: {config}")

    # Run comparison
    results = compare_all_strategies()

    # Print report
    print_comparison_report(results)

    # Generate recommendations
    recommendations = generate_recommendations(results)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    for rec in recommendations:
        print(f"{rec}")

    print("\n" + "=" * 80)
    print("IMPLEMENTATION NOTES")
    print("=" * 80)
    print("1. Late chunking preserves better context but may be slower")
    print("2. Hybrid strategies balance performance and quality")
    print("3. Task-specific models (Jina v4) can be optimized for retrieval")
    print("4. Consider your use case: speed vs. quality trade-offs")
    print("5. Japanese text benefits from language-aware chunking")


if __name__ == "__main__":
    main()
