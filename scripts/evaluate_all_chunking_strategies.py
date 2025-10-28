#!/usr/bin/env python3
"""
Comprehensive chunking strategy comparison and evaluation script.

This script tests all available chunking strategies across all supported embedding models,
documenting their capabilities, limitations, and performance characteristics.

Usage:
    python scripts/evaluate_all_chunking_strategies.py [--models MODEL1,MODEL2] [--strategies STRATEGY1,STRATEGY2] [--output results.json]

Features:
- Automatic model capability detection
- Fallback mechanism testing
- Performance benchmarking
- Japanese-specific evaluation
- Detailed capability documentation
- Alternative strategy recommendations
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling_japanese_books.config import config
from docling_japanese_books.embedding_evaluation import MultiStrategyEmbeddingEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChunkingStrategyAnalyzer:
    """
    Comprehensive analyzer for chunking strategies across embedding models.

    This class provides detailed analysis of which strategies work with which models,
    performance characteristics, and recommendations for different use cases.
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.evaluator = MultiStrategyEmbeddingEvaluator()
        self.config = config

        # Sample Japanese document for testing
        self.sample_document = self._load_sample_document()

        # All available models to test
        self.available_models = [
            "BAAI/bge-m3",
            "Snowflake/snowflake-arctic-embed-l-v2.0",
            "jinaai/jina-embeddings-v4",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

        # All available strategies to test
        self.available_strategies = ["late", "traditional", "hybrid", "hierarchical"]

    def _load_sample_document(self) -> str:
        """Load sample Japanese document for evaluation."""
        # Try to load from processed documents first
        processed_dir = (
            Path(self.config.output.output_base_dir)
            / self.config.output.processed_output_dir
        )

        if processed_dir.exists():
            for file_path in processed_dir.glob("*.md"):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                    if len(content) > 1000:
                        logger.info(f"Using sample document: {file_path.name}")
                        return content[:5000]  # Limit to first 5000 chars
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
                    continue

        # Fallback to embedded sample
        return """
        コンピュータサイエンスにおける自然言語処理（しぜんげんごしょり、Natural Language Processing、NLP）は、
        人間が日常的に使っている自然言語をコンピュータに処理させる一連の技術である。
        自然言語処理は言語学、コンピュータ科学、人工知能の学際的な分野である。
        
        近年、深層学習の発展により、自然言語処理技術は大幅に向上した。
        特に、Transformerアーキテクチャの登場は、機械翻訳、文書要約、質問応答などの
        タスクにおいて革命的な改善をもたらした。
        
        日本語の自然言語処理は、特有の課題を持つ。漢字、ひらがな、カタカナという
        三つの文字体系の混在、語順の柔軟性、敬語システムの複雑さなどが挙げられる。
        これらの特徴により、日本語専用の前処理技術やモデルの開発が重要である。
        
        埋め込み（エンベッディング）技術は、テキストを数値ベクトルに変換する技術であり、
        検索、分類、クラスタリングなどの下流タスクの基盤となっている。
        BGE-M3やJina Embeddings v4などの多言語対応モデルは、
        日本語テキストの処理において優れた性能を示している。
        
        Late Chunkingは、従来のchunk-firstアプローチとは異なり、
        まず文書全体をエンベッディングしてからチャンクに分割する手法である。
        この手法により、チャンク間のコンテキストをより良く保持することができ、
        特に日本語のような複雑な文法構造を持つ言語において有効である。
        
        量子化対応訓練（Quantization-Aware Training）は、
        モデルの計算効率を向上させる手法として注目されている。
        Jina Embeddings v4のような最新モデルでは、
        この技術により品質を保ちながらメモリ使用量を削減している。
        """

    def analyze_model_capabilities(
        self, models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze capabilities of each model with different chunking strategies.

        Args:
            models: List of models to analyze, or None for all models

        Returns:
            Comprehensive capability analysis
        """
        if models is None:
            models = self.available_models

        logger.info(f"Analyzing capabilities for {len(models)} models...")

        analysis = {
            "models": {},
            "strategy_compatibility": {},
            "performance_summary": {},
            "recommendations": {},
            "alternatives": {},
            "timestamp": time.time(),
        }

        for model_name in models:
            logger.info(f"\\n{'=' * 60}")
            logger.info(f"ANALYZING MODEL: {model_name}")
            logger.info(f"{'=' * 60}")

            model_analysis = {
                "model_name": model_name,
                "supported_strategies": [],
                "failed_strategies": [],
                "strategy_results": {},
                "best_strategy": None,
                "alternatives": [],
                "limitations": [],
                "recommendations": [],
            }

            # Test each strategy with this model
            for strategy in self.available_strategies:
                logger.info(f"Testing {strategy} chunking...")

                try:
                    # Test if we can create the strategy
                    chunker, is_fallback, fallback_reason = (
                        self.evaluator.create_chunking_strategy(model_name, strategy)
                    )

                    # Evaluate performance
                    start_time = time.time()
                    chunks, embeddings = chunker.process_document(
                        self.sample_document, 400
                    )
                    processing_time = time.time() - start_time

                    # Calculate basic metrics
                    num_chunks = len(chunks)
                    avg_chunk_length = sum(len(chunk) for chunk in chunks) / max(
                        num_chunks, 1
                    )
                    embedding_dim = len(embeddings[0]) if embeddings else 0

                    # Context preservation (simplified)
                    context_score = 0.0
                    if len(embeddings) > 1:
                        import numpy as np

                        similarities = []
                        for i in range(len(embeddings) - 1):
                            if len(embeddings[i]) > 0 and len(embeddings[i + 1]) > 0:
                                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                                    np.linalg.norm(embeddings[i])
                                    * np.linalg.norm(embeddings[i + 1])
                                )
                                similarities.append(sim)
                        context_score = np.mean(similarities) if similarities else 0.0

                    result = {
                        "status": "success",
                        "is_fallback": is_fallback,
                        "fallback_reason": fallback_reason,
                        "processing_time": processing_time,
                        "num_chunks": num_chunks,
                        "avg_chunk_length": int(avg_chunk_length),
                        "embedding_dimension": embedding_dim,
                        "context_preservation_score": float(context_score),
                        "chunks_sample": chunks[:2]
                        if chunks
                        else [],  # First 2 chunks for inspection
                    }

                    if is_fallback:
                        model_analysis["alternatives"].append(
                            {
                                "requested": strategy,
                                "actual": chunker.strategy_used
                                if hasattr(chunker, "strategy_used")
                                else "unknown",
                                "reason": fallback_reason,
                            }
                        )
                        logger.warning(
                            f"  ⚠️  {strategy}: Used fallback - {fallback_reason}"
                        )
                    else:
                        model_analysis["supported_strategies"].append(strategy)
                        logger.info(
                            f"  ✅ {strategy}: Success ({processing_time:.2f}s, {num_chunks} chunks)"
                        )

                    model_analysis["strategy_results"][strategy] = result

                except Exception as e:
                    logger.error(f"  ❌ {strategy}: Failed - {str(e)}")
                    model_analysis["failed_strategies"].append(strategy)
                    model_analysis["strategy_results"][strategy] = {
                        "status": "failed",
                        "error": str(e),
                        "limitations": [f"Cannot use {strategy} with {model_name}"],
                    }

            # Determine best strategy for this model
            successful_results = [
                (strategy, result)
                for strategy, result in model_analysis["strategy_results"].items()
                if result.get("status") == "success"
                and not result.get("is_fallback", False)
            ]

            if successful_results:
                # Sort by context preservation score, then by speed
                best_strategy, best_result = max(
                    successful_results,
                    key=lambda x: (
                        x[1].get("context_preservation_score", 0),
                        -x[1].get("processing_time", 999),
                    ),
                )
                model_analysis["best_strategy"] = {
                    "strategy": best_strategy,
                    "metrics": best_result,
                }

                logger.info(f"  🏆 Best strategy: {best_strategy}")

            # Generate model-specific recommendations
            model_analysis["recommendations"] = self._generate_model_recommendations(
                model_name, model_analysis
            )

            # Document limitations
            model_analysis["limitations"] = self._document_model_limitations(
                model_name, model_analysis
            )

            analysis["models"][model_name] = model_analysis

        # Generate cross-model analysis
        analysis["strategy_compatibility"] = self._analyze_strategy_compatibility(
            analysis["models"]
        )
        analysis["performance_summary"] = self._generate_performance_summary(
            analysis["models"]
        )
        analysis["recommendations"] = self._generate_global_recommendations(analysis)
        analysis["alternatives"] = self._document_alternatives(analysis)

        return analysis

    def _generate_model_recommendations(
        self, model_name: str, model_analysis: Dict
    ) -> List[str]:
        """Generate recommendations for specific model."""
        recommendations = []

        if model_analysis["best_strategy"]:
            best = model_analysis["best_strategy"]
            recommendations.append(
                f"Use {best['strategy']} chunking for optimal performance "
                f"(context score: {best['metrics'].get('context_preservation_score', 0):.3f})"
            )

        # Model-specific advice
        if "bge-m3" in model_name.lower():
            recommendations.extend(
                [
                    "BGE-M3 excels with late chunking for Japanese context preservation",
                    "Consider hybrid chunking for production use with fallback safety",
                    "Optimal chunk size: 400 characters for Japanese text",
                ]
            )
        elif "jina-embeddings-v4" in model_name.lower():
            recommendations.extend(
                [
                    "Jina v4 benefits from task-specific optimization (task='retrieval')",
                    "Quantization-aware training provides memory efficiency",
                    "Hybrid chunking balances quality and performance",
                ]
            )
        elif "snowflake" in model_name.lower():
            recommendations.extend(
                [
                    "Snowflake Arctic optimized for speed with traditional chunking",
                    "High-quality embeddings suitable for multilingual content",
                    "Consider for high-throughput scenarios",
                ]
            )
        else:
            recommendations.append(
                "Use traditional chunking for reliable baseline performance"
            )

        return recommendations

    def _document_model_limitations(
        self, model_name: str, model_analysis: Dict
    ) -> List[str]:
        """Document known limitations for model."""
        limitations = []

        failed_strategies = model_analysis.get("failed_strategies", [])
        if failed_strategies:
            limitations.append(
                f"Cannot use {', '.join(failed_strategies)} chunking strategies"
            )

        # Model-specific limitations
        if "bge-m3" in model_name.lower():
            limitations.extend(
                [
                    "FlagEmbedding API may not expose token-level embeddings directly",
                    "Late chunking implementation may fall back to sentence-level",
                    "Higher memory usage with full document embedding",
                ]
            )
        elif "jina-embeddings-v4" in model_name.lower():
            limitations.extend(
                [
                    "No native late chunking support (approximation possible)",
                    "Requires task specification for optimal performance",
                    "Newer model with less Japanese-specific optimization",
                ]
            )
        elif "snowflake" in model_name.lower():
            limitations.extend(
                [
                    "No token-level embedding access for late chunking",
                    "Primarily English-optimized (decent multilingual support)",
                    "May require adaptation for Japanese-specific use cases",
                ]
            )
        else:
            limitations.extend(
                [
                    "Limited embedding dimensions (384 for MiniLM)",
                    "English-focused model with limited Japanese optimization",
                    "No advanced chunking strategy support",
                ]
            )

        return limitations

    def _analyze_strategy_compatibility(self, models: Dict) -> Dict[str, Any]:
        """Analyze which strategies work with which models."""
        compatibility = {}

        for strategy in self.available_strategies:
            compatible_models = []
            incompatible_models = []
            fallback_models = []

            for model_name, model_data in models.items():
                if strategy in model_data.get("supported_strategies", []):
                    compatible_models.append(model_name)
                elif strategy in model_data.get("failed_strategies", []):
                    incompatible_models.append(model_name)
                else:
                    # Check if it's a fallback case
                    strategy_result = model_data.get("strategy_results", {}).get(
                        strategy, {}
                    )
                    if strategy_result.get("is_fallback"):
                        fallback_models.append(model_name)
                    else:
                        incompatible_models.append(model_name)

            compatibility[strategy] = {
                "fully_compatible": compatible_models,
                "fallback_available": fallback_models,
                "incompatible": incompatible_models,
                "compatibility_rate": len(compatible_models) / len(models)
                if models
                else 0,
                "description": self.evaluator.strategy_definitions.get(
                    strategy, {}
                ).get("description", "Unknown strategy"),
            }

        return compatibility

    def _generate_performance_summary(self, models: Dict) -> Dict[str, Any]:
        """Generate performance summary across all models."""
        summary = {
            "fastest_combinations": [],
            "best_context_preservation": [],
            "most_reliable": [],
            "resource_efficient": [],
        }

        all_results = []
        for model_name, model_data in models.items():
            for strategy, result in model_data.get("strategy_results", {}).items():
                if result.get("status") == "success":
                    all_results.append(
                        {
                            "model": model_name,
                            "strategy": strategy,
                            "is_fallback": result.get("is_fallback", False),
                            **result,
                        }
                    )

        if all_results:
            # Fastest combinations
            fastest = sorted(all_results, key=lambda x: x.get("processing_time", 999))[
                :3
            ]
            summary["fastest_combinations"] = [
                f"{r['model']} + {r['strategy']} ({r['processing_time']:.2f}s)"
                for r in fastest
            ]

            # Best context preservation
            best_context = sorted(
                all_results,
                key=lambda x: x.get("context_preservation_score", 0),
                reverse=True,
            )[:3]
            summary["best_context_preservation"] = [
                f"{r['model']} + {r['strategy']} (score: {r['context_preservation_score']:.3f})"
                for r in best_context
            ]

            # Most reliable (non-fallback)
            reliable = [r for r in all_results if not r.get("is_fallback", False)]
            summary["most_reliable"] = [
                f"{r['model']} + {r['strategy']}" for r in reliable[:5]
            ]

            # Resource efficient (smaller embeddings, faster processing)
            efficient = sorted(
                all_results,
                key=lambda x: (
                    x.get("processing_time", 999),
                    -x.get("embedding_dimension", 0),
                ),
            )[:3]
            summary["resource_efficient"] = [
                f"{r['model']} + {r['strategy']} ({r['embedding_dimension']}D, {r['processing_time']:.2f}s)"
                for r in efficient
            ]

        return summary

    def _generate_global_recommendations(self, analysis: Dict) -> Dict[str, List[str]]:
        """Generate global recommendations based on use cases."""
        return {
            "production_quality": [
                "Use BGE-M3 with late chunking for best Japanese context preservation",
                "Implement hybrid chunking with fallback mechanisms for reliability",
                "Set chunk size to 400 characters for Japanese content",
            ],
            "production_speed": [
                "Use Snowflake Arctic with traditional chunking for high throughput",
                "Consider Jina v4 with hybrid chunking for balanced performance",
                "Implement caching for frequently accessed embeddings",
            ],
            "development_testing": [
                "Use all-MiniLM-L6-v2 with traditional chunking for quick iteration",
                "Test with multiple strategies to understand performance trade-offs",
                "Start with hybrid chunking for production planning",
            ],
            "research_experimentation": [
                "Compare all models with hierarchical chunking for query diversity",
                "Evaluate late chunking approximations for non-supporting models",
                "Measure context preservation across different document types",
            ],
        }

    def _document_alternatives(self, analysis: Dict) -> Dict[str, Any]:
        """Document alternative approaches when preferred strategies fail."""
        alternatives = {}

        for model_name, model_data in analysis["models"].items():
            model_alternatives = []

            # Document fallback cases
            for alt in model_data.get("alternatives", []):
                model_alternatives.append(
                    {
                        "scenario": f"When {alt['requested']} is requested",
                        "alternative": alt["actual"],
                        "reason": alt["reason"],
                        "impact": "May have different performance characteristics",
                    }
                )

            # Document failed strategies and their alternatives
            for failed_strategy in model_data.get("failed_strategies", []):
                # Find working alternatives
                working_strategies = model_data.get("supported_strategies", [])
                if working_strategies:
                    best_alternative = working_strategies[0]  # First working strategy
                    model_alternatives.append(
                        {
                            "scenario": f"When {failed_strategy} fails completely",
                            "alternative": best_alternative,
                            "reason": f"{failed_strategy} not supported by {model_name}",
                            "impact": "Switch to different chunking approach",
                        }
                    )

            alternatives[model_name] = model_alternatives

        return alternatives

    def generate_report(self, analysis: Dict, output_path: Optional[str] = None) -> str:
        """Generate comprehensive human-readable report."""
        report_lines = [
            "# Comprehensive Chunking Strategy Analysis Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"Analyzed {len(analysis['models'])} embedding models with {len(self.available_strategies)} chunking strategies.",
            "This report documents capabilities, limitations, and recommendations for each combination.",
            "",
        ]

        # Model-by-model analysis
        report_lines.append("## Model Analysis")
        report_lines.append("")

        for model_name, model_data in analysis["models"].items():
            report_lines.extend(
                [
                    f"### {model_name}",
                    "",
                    f"**Supported Strategies:** {', '.join(model_data['supported_strategies']) or 'None'}",
                    f"**Failed Strategies:** {', '.join(model_data['failed_strategies']) or 'None'}",
                    "",
                ]
            )

            if model_data.get("best_strategy"):
                best = model_data["best_strategy"]
                report_lines.extend(
                    [
                        f"**Best Strategy:** {best['strategy']}",
                        f"- Processing Time: {best['metrics']['processing_time']:.2f}s",
                        f"- Chunks Generated: {best['metrics']['num_chunks']}",
                        f"- Context Score: {best['metrics']['context_preservation_score']:.3f}",
                        "",
                    ]
                )

            # Recommendations
            if model_data.get("recommendations"):
                report_lines.append("**Recommendations:**")
                for rec in model_data["recommendations"]:
                    report_lines.append(f"- {rec}")
                report_lines.append("")

            # Limitations
            if model_data.get("limitations"):
                report_lines.append("**Limitations:**")
                for limit in model_data["limitations"]:
                    report_lines.append(f"- {limit}")
                report_lines.append("")

        # Strategy compatibility
        report_lines.extend(
            [
                "## Strategy Compatibility Matrix",
                "",
                "| Strategy | Compatible Models | Compatibility Rate | Description |",
                "|----------|------------------|-------------------|-------------|",
            ]
        )

        for strategy, compat_data in analysis["strategy_compatibility"].items():
            compatible = ", ".join(compat_data["fully_compatible"]) or "None"
            rate = f"{compat_data['compatibility_rate']:.1%}"
            desc = (
                compat_data["description"][:50] + "..."
                if len(compat_data["description"]) > 50
                else compat_data["description"]
            )
            report_lines.append(f"| {strategy} | {compatible} | {rate} | {desc} |")

        report_lines.append("")

        # Performance summary
        perf = analysis["performance_summary"]
        report_lines.extend(["## Performance Summary", "", "**Fastest Combinations:**"])
        for fast in perf["fastest_combinations"]:
            report_lines.append(f"- {fast}")

        report_lines.extend(["", "**Best Context Preservation:**"])
        for context in perf["best_context_preservation"]:
            report_lines.append(f"- {context}")

        # Recommendations by use case
        report_lines.extend(["", "## Recommendations by Use Case", ""])

        for use_case, recs in analysis["recommendations"].items():
            report_lines.extend([f"### {use_case.replace('_', ' ').title()}", ""])
            for rec in recs:
                report_lines.append(f"- {rec}")
            report_lines.append("")

        # Alternatives documentation
        report_lines.extend(
            [
                "## Alternative Strategies and Fallbacks",
                "",
                "When preferred strategies are not available, the following alternatives are used:",
                "",
            ]
        )

        for model_name, alternatives in analysis["alternatives"].items():
            if alternatives:
                report_lines.extend([f"### {model_name}", ""])
                for alt in alternatives:
                    report_lines.extend(
                        [
                            f"**{alt['scenario']}:**",
                            f"- Alternative: {alt['alternative']}",
                            f"- Reason: {alt['reason']}",
                            f"- Impact: {alt['impact']}",
                            "",
                        ]
                    )

        report_content = "\\n".join(report_lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_path}")

        return report_content


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive chunking strategy analysis for embedding models"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to test (default: all)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of strategies to test (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunking_strategy_analysis.json",
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="chunking_strategy_report.md",
        help="Output file for markdown report",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with minimal models"
    )

    args = parser.parse_args()

    analyzer = ChunkingStrategyAnalyzer()

    # Parse model list
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.quick:
        models = ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]
    else:
        models = None  # Use all models

    # Parse strategy list
    if args.strategies:
        # Note: This script analyzes model capabilities, strategies are tested automatically
        logger.info(f"Strategy filter specified: {args.strategies}")

    logger.info("Starting comprehensive chunking strategy analysis...")
    logger.info(f"Models to test: {models or 'all available'}")

    try:
        # Run the analysis
        analysis = analyzer.analyze_model_capabilities(models)

        # Save JSON results
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Analysis results saved to: {args.output}")

        # Generate and save report
        report = analyzer.generate_report(analysis, args.report)

        # Print summary to console
        print("\\n" + "=" * 80)
        print("CHUNKING STRATEGY ANALYSIS SUMMARY")
        print("=" * 80)

        for model_name, model_data in analysis["models"].items():
            print(f"\\n{model_name}:")
            print(
                f"  Supported: {', '.join(model_data['supported_strategies']) or 'None'}"
            )
            print(f"  Failed: {', '.join(model_data['failed_strategies']) or 'None'}")
            if model_data.get("best_strategy"):
                best = model_data["best_strategy"]
                print(
                    f"  Best: {best['strategy']} (context: {best['metrics']['context_preservation_score']:.3f})"
                )

        print(f"\\nDetailed results: {args.output}")
        print(f"Human-readable report: {args.report}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
