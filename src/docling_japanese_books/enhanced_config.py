"""
Enhanced configuration for chunking strategies.

This extends the existing config.py to support multiple chunking strategies
and model-specific optimizations.
"""

from typing import Any

from pydantic import Field

from .config import ChunkingConfig as BaseChunkingConfig


class EnhancedChunkingConfig(BaseChunkingConfig):
    """Extended chunking configuration with support for multiple strategies."""

    # Available models and their optimal configurations
    model_configurations: dict[str, dict[str, Any]] = Field(
        default={
            "BAAI/bge-m3": {
                "supports_late_chunking": True,
                "optimal_chunk_size": 400,
                "embedding_dim": 1024,
                "recommended_strategy": "late",
                "task": None,
                "notes": "Multilingual model optimized for Japanese content",
            },
            "Snowflake/snowflake-arctic-embed-l-v2.0": {
                "supports_late_chunking": False,  # Can be implemented with custom approach
                "optimal_chunk_size": 512,
                "embedding_dim": 1024,
                "recommended_strategy": "hybrid",
                "task": None,
                "notes": "High-quality embeddings, good for English and some multilingual content",
            },
            "jinaai/jina-embeddings-v4": {
                "supports_late_chunking": False,  # Can be implemented with custom approach
                "optimal_chunk_size": 512,
                "embedding_dim": 1024,
                "recommended_strategy": "hybrid",
                "task": "retrieval",  # Can be 'retrieval', 'text-matching', 'code'
                "notes": "Quantization-aware training, task-specific optimization",
            },
            "sentence-transformers/all-MiniLM-L6-v2": {
                "supports_late_chunking": False,
                "optimal_chunk_size": 384,
                "embedding_dim": 384,
                "recommended_strategy": "traditional",
                "task": None,
                "notes": "Lightweight model, good for development and testing",
            },
        },
        description="Model-specific configurations and capabilities",
    )

    # Strategy configurations
    available_strategies: dict[str, dict[str, Any]] = Field(
        default={
            "late": {
                "name": "Late Chunking",
                "description": "Embed full document first, then chunk and pool token embeddings",
                "benefits": [
                    "Better context preservation",
                    "Cross-chunk relationships",
                    "Japanese grammar awareness",
                ],
                "drawbacks": [
                    "Slower processing",
                    "Higher memory usage",
                    "Model-dependent availability",
                ],
                "best_for": "Complex queries requiring contextual understanding",
            },
            "traditional": {
                "name": "Traditional Chunking",
                "description": "Chunk first, then embed each chunk independently",
                "benefits": [
                    "Fast processing",
                    "Low memory usage",
                    "Works with all models",
                ],
                "drawbacks": [
                    "No cross-chunk context",
                    "Potential information loss at boundaries",
                ],
                "best_for": "Simple keyword searches and high-volume processing",
            },
            "hybrid": {
                "name": "Hybrid Chunking",
                "description": "Combines late chunking (where possible) with traditional fallback",
                "benefits": [
                    "Model adaptability",
                    "Balanced performance",
                    "Graceful degradation",
                ],
                "drawbacks": [
                    "Complex implementation",
                    "Strategy-dependent performance",
                ],
                "best_for": "Production systems with mixed query types",
            },
            "hierarchical": {
                "name": "Hierarchical Chunking",
                "description": "Multiple chunk sizes for different query granularities",
                "benefits": [
                    "Query-type optimization",
                    "Better recall",
                    "Flexible retrieval",
                ],
                "drawbacks": [
                    "Storage overhead",
                    "Complex retrieval logic",
                    "Higher processing cost",
                ],
                "best_for": "Advanced search systems with diverse query patterns",
            },
        },
        description="Available chunking strategies and their characteristics",
    )

    # Current strategy selection
    primary_strategy: str = Field(
        default="hybrid", description="Primary chunking strategy to use"
    )

    fallback_strategy: str = Field(
        default="traditional", description="Fallback strategy when primary fails"
    )

    # Enhanced Japanese text processing
    japanese_specific_settings: dict[str, Any] = Field(
        default={
            "enable_japanese_chunking": True,
            "sentence_patterns": [
                r"[。！？]+",  # Standard Japanese endings
                r"[\.!?]+",  # Western punctuation
                r"」[。！？]*",  # Quote endings
                r"』[。！？]*",  # Book quote endings
                r"[。！？]*\n\n",  # Paragraph breaks
            ],
            "preserve_honorifics": True,
            "merge_short_chunks": True,
            "min_chunk_chars": 50,
            "optimal_chunk_chars": 400,
            "max_chunk_chars": 800,
        },
        description="Japanese-specific text processing settings",
    )

    # Performance optimization
    performance_settings: dict[str, Any] = Field(
        default={
            "enable_caching": True,
            "cache_embeddings": True,
            "batch_processing": True,
            "max_batch_size": 32,
            "memory_limit_mb": 2048,
            "use_gpu_if_available": True,
        },
        description="Performance optimization settings",
    )

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get configuration for specific model."""
        return self.model_configurations.get(
            model_name,
            {
                "supports_late_chunking": False,
                "optimal_chunk_size": 500,
                "embedding_dim": 768,
                "recommended_strategy": "traditional",
                "task": None,
                "notes": "Unknown model, using default settings",
            },
        )

    def get_strategy_info(self, strategy: str) -> dict[str, Any]:
        """Get information about chunking strategy."""
        return self.available_strategies.get(
            strategy,
            {
                "name": "Unknown Strategy",
                "description": "Strategy not found in configuration",
                "benefits": [],
                "drawbacks": ["Unknown behavior"],
                "best_for": "Not recommended",
            },
        )

    def recommend_strategy_for_model(self, model_name: str) -> str:
        """Recommend best chunking strategy for given model."""
        model_config = self.get_model_config(model_name)
        return model_config.get("recommended_strategy", self.primary_strategy)

    def get_optimal_chunk_size(self, model_name: str) -> int:
        """Get optimal chunk size for model."""
        model_config = self.get_model_config(model_name)
        return model_config.get("optimal_chunk_size", self.max_chunk_tokens)

    def supports_late_chunking(self, model_name: str) -> bool:
        """Check if model supports late chunking."""
        model_config = self.get_model_config(model_name)
        return model_config.get("supports_late_chunking", False)


# Usage example configuration
CHUNKING_RECOMMENDATIONS = {
    "development": {
        "models": ["sentence-transformers/all-MiniLM-L6-v2"],
        "strategy": "traditional",
        "chunk_size": 384,
        "rationale": "Fast iteration, lightweight",
    },
    "production_balanced": {
        "models": ["BAAI/bge-m3", "jinaai/jina-embeddings-v4"],
        "strategy": "hybrid",
        "chunk_size": 400,
        "rationale": "Good balance of quality and performance",
    },
    "production_quality": {
        "models": ["BAAI/bge-m3"],
        "strategy": "late",
        "chunk_size": 400,
        "rationale": "Maximum quality for Japanese content",
    },
    "production_speed": {
        "models": ["Snowflake/snowflake-arctic-embed-l-v2.0"],
        "strategy": "traditional",
        "chunk_size": 512,
        "rationale": "Optimized for high-throughput scenarios",
    },
    "research": {
        "models": [
            "BAAI/bge-m3",
            "jinaai/jina-embeddings-v4",
            "Snowflake/snowflake-arctic-embed-l-v2.0",
        ],
        "strategy": "hierarchical",
        "chunk_size": [200, 400, 800],
        "rationale": "Comprehensive evaluation and comparison",
    },
}
