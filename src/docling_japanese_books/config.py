"""Configuration settings for the document processing pipeline with cloud support."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DoclingConfig(BaseModel):
    """Document processing configuration optimized for Japanese content."""

    artifacts_path: str = Field(
        default=".models",
        description="Local model cache directory",
    )
    enable_ocr: bool = Field(default=True, description="OCR for text extraction")
    enable_vision: bool = Field(
        default=False,
        description="Vision model for image descriptions",
    )
    do_table_structure: bool = Field(
        default=True, description="Extract table structure"
    )
    do_cell_matching: bool = Field(
        default=True, description="Match table cells in PDFs"
    )
    generate_page_images: bool = Field(
        default=True,
        description="Generate page images for output",
    )
    vision_model: str = Field(
        default="granite",
        description="Vision model type",
    )
    vision_model_repo_id: str = Field(
        default="ibm-granite/granite-vision-3.3-2b",
        description="HuggingFace model ID for image descriptions",
    )
    vision_prompt: str = Field(
        default="Describe this image in detail, paying special attention to any Japanese text, characters, or cultural elements. Include information about layout, visual elements, and any text content visible in the image.",
        description="Vision model prompt optimized for Japanese content",
    )
    images_scale: float = Field(
        default=2.0, description="Image processing scale factor"
    )
    max_file_size_mb: int = Field(
        default=100, description="Maximum file size limit (MB)"
    )
    max_num_pages: int = Field(default=1000, description="Maximum pages per document")
    thread_count: int = Field(default=4, description="Processing thread count")
    supported_formats: list[str] = Field(
        default=[
            ".pdf",
            ".docx",
            ".pptx",
            ".html",
            ".htm",
            ".md",
            ".txt",
            ".png",
            ".jpg",
            ".jpeg",
        ],
        description="Supported file extensions",
    )


class DatabaseConfig(BaseModel):
    """Vector database configuration for Milvus Lite and Zilliz Cloud."""

    database_type: str = Field(default="milvus", description="Vector database type")
    milvus_uri: str = Field(
        default="http://localhost:19530",
        description="Milvus Docker host URI (default port)",
    )
    zilliz_cloud_uri: str = Field(
        default_factory=lambda: os.getenv("ZILLIZ_CLOUD_URI", ""),
        description="Zilliz Cloud endpoint URI",
    )
    zilliz_api_key: str = Field(
        default_factory=lambda: os.getenv("ZILLIZ_API_KEY", ""),
        description="Zilliz Cloud API key",
    )
    zilliz_cluster_id: str = Field(
        default_factory=lambda: os.getenv("ZILLIZ_CLUSTER_ID", ""),
        description="Zilliz Cloud cluster identifier",
    )
    deployment_mode: str = Field(
        default_factory=lambda: os.getenv("MILVUS_DEPLOYMENT_MODE", "docker"),
        description="Database deployment mode: 'local', 'docker', or 'cloud'",
    )
    collection_name: str = Field(
        default="docling_japanese_books", description="Vector database collection name"
    )
    embedding_dimension: int = Field(
        default=1024,
        description="BGE-M3 embedding vector dimension",
    )
    connection_timeout: int = Field(
        default=30, description="Database connection timeout (seconds)"
    )
    consistency_level: str = Field(
        default="Strong",
        description="Milvus consistency level",
    )

    def get_connection_uri(self) -> str:
        """Return connection URI for current deployment mode."""
        if self.deployment_mode == "cloud":
            if not self.zilliz_cloud_uri:
                raise ValueError(
                    "Zilliz Cloud URI is required for cloud deployment mode"
                )
            return self.zilliz_cloud_uri
        else:
            return self.milvus_uri

    def get_connection_params(self) -> Dict[str, str]:
        """Get connection parameters for Milvus based on deployment mode."""
        if self.deployment_mode == "cloud":
            return {
                "uri": self.zilliz_cloud_uri,
                "token": self.zilliz_api_key,
            }
        else:
            return {"uri": self.get_connection_uri()}


class ChunkingConfig(BaseModel):
    """
    Enhanced text chunking and embedding configuration for Japanese documents.

    This configuration supports multiple chunking strategies per model with automatic
    fallback mechanisms and model-specific optimizations.
    """

    tokenizer_model: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="Granite Docling model for document tokenization",
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Primary embedding model - BGE-M3 multilingual embedding model",
    )

    # Enhanced chunking strategy configuration
    chunking_strategy: str = Field(
        default="auto",
        description="""
        Chunking strategy selection:
        - 'auto': Automatically select best strategy per model
        - 'late': Late chunking (embed full document, then chunk)
        - 'traditional': Chunk first, then embed each chunk
        - 'hybrid': Model-adaptive strategy with fallbacks
        - 'hierarchical': Multiple chunk sizes for different queries
        """,
    )

    # Model-specific configurations
    model_specific_settings: Dict[str, Any] = Field(
        default={
            "BAAI/bge-m3": {
                "preferred_strategy": "late",
                "fallback_strategies": ["hybrid", "traditional"],
                "optimal_chunk_size": 400,
                "supports_late_chunking": True,
                "task": None,
                "notes": "Multilingual model optimized for context preservation",
            },
            "jinaai/jina-embeddings-v4": {
                "preferred_strategy": "hybrid",
                "fallback_strategies": ["traditional"],
                "optimal_chunk_size": 512,
                "supports_late_chunking": False,
                "task": "retrieval",
                "notes": "Quantization-aware with task-specific optimization",
            },
            "Snowflake/snowflake-arctic-embed-l-v2.0": {
                "preferred_strategy": "traditional",
                "fallback_strategies": ["hybrid"],
                "optimal_chunk_size": 512,
                "supports_late_chunking": False,
                "task": None,
                "notes": "High-quality embeddings optimized for speed",
            },
            "sentence-transformers/all-MiniLM-L6-v2": {
                "preferred_strategy": "traditional",
                "fallback_strategies": ["hybrid"],
                "optimal_chunk_size": 384,
                "supports_late_chunking": False,
                "task": None,
                "notes": "Lightweight baseline model",
            },
        },
        description="Model-specific chunking preferences and capabilities",
    )

    # Strategy-specific settings
    strategy_settings: Dict[str, Any] = Field(
        default={
            "late": {
                "description": "Embed full document first, then chunk and pool tokens",
                "memory_intensive": True,
                "processing_time": "slow",
                "context_preservation": "excellent",
                "best_for": [
                    "Japanese text",
                    "contextual queries",
                    "complex documents",
                ],
            },
            "traditional": {
                "description": "Chunk document first, then embed each chunk",
                "memory_intensive": False,
                "processing_time": "fast",
                "context_preservation": "limited",
                "best_for": ["keyword search", "high throughput", "simple queries"],
            },
            "hybrid": {
                "description": "Adaptive strategy using best available method per model",
                "memory_intensive": "variable",
                "processing_time": "balanced",
                "context_preservation": "good",
                "best_for": ["production systems", "mixed query types", "reliability"],
            },
            "hierarchical": {
                "description": "Multiple chunk sizes for different query granularities",
                "memory_intensive": True,
                "processing_time": "slow",
                "context_preservation": "variable",
                "best_for": ["advanced search", "diverse queries", "research systems"],
            },
        },
        description="Chunking strategy characteristics and use cases",
    )

    # Legacy settings (maintained for backward compatibility)
    max_chunk_tokens: int = Field(
        default=512, description="Maximum tokens per chunk (legacy)"
    )
    chunk_overlap: int = Field(default=50, description="Token overlap between chunks")
    min_chunk_length: int = Field(
        default=20, description="Minimum chunk length (tokens)"
    )
    merge_list_items: bool = Field(
        default=True, description="Merge list items during chunking"
    )
    use_late_chunking: bool = Field(
        default=True,
        description="Enable Late Chunking for better context preservation (legacy)",
    )
    merge_peers: bool = Field(
        default=True, description="Merge peer chunks in hybrid strategy"
    )

    # Japanese-specific enhancements
    japanese_optimization: Dict[str, Any] = Field(
        default={
            "enable_japanese_chunking": True,
            "respect_sentence_boundaries": True,
            "preserve_honorifics": True,
            "handle_mixed_scripts": True,
            "sentence_patterns": [
                r"[。！？]+",  # Standard Japanese endings
                r"[\.!?]+",  # Western punctuation
                r"」[。！？]*",  # Quote endings
                r"』[。！？]*",  # Book quote endings
            ],
            "min_chunk_chars": 50,
            "optimal_chunk_chars": 400,
            "max_chunk_chars": 800,
        },
        description="Japanese-specific text processing optimizations",
    )

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model."""
        return self.model_specific_settings.get(
            model_name,
            {
                "preferred_strategy": "traditional",
                "fallback_strategies": ["hybrid"],
                "optimal_chunk_size": 500,
                "supports_late_chunking": False,
                "task": None,
                "notes": "Default configuration for unknown model",
            },
        )

    def get_optimal_chunk_size(self, model_name: str) -> int:
        """Get optimal chunk size for model."""
        model_config = self.get_model_config(model_name)
        return model_config.get("optimal_chunk_size", self.max_chunk_tokens)

    def get_preferred_strategy(self, model_name: str) -> str:
        """Get preferred chunking strategy for model."""
        if self.chunking_strategy == "auto":
            model_config = self.get_model_config(model_name)
            return model_config.get("preferred_strategy", "traditional")
        return self.chunking_strategy

    def get_fallback_strategies(self, model_name: str) -> List[str]:
        """Get fallback strategies for model."""
        model_config = self.get_model_config(model_name)
        return model_config.get("fallback_strategies", ["traditional"])

    def supports_late_chunking(self, model_name: str) -> bool:
        """Check if model supports late chunking."""
        model_config = self.get_model_config(model_name)
        return model_config.get("supports_late_chunking", False)

    def get_task_for_model(self, model_name: str) -> Optional[str]:
        """Get task specification for task-aware models."""
        model_config = self.get_model_config(model_name)
        return model_config.get("task")


class OutputConfig(BaseModel):
    """File output format and directory configuration."""

    output_formats: list[str] = Field(
        default=["json", "markdown", "jsonl"],
        description="Document export formats",
    )
    output_base_dir: str = Field(
        default="./output", description="Base output directory"
    )
    raw_output_dir: str = Field(
        default="raw", description="Raw document output subdirectory"
    )
    processed_output_dir: str = Field(
        default="processed", description="Processed document subdirectory"
    )
    chunks_output_dir: str = Field(
        default="chunks", description="Text chunks subdirectory"
    )
    images_output_dir: str = Field(
        default="images", description="Extracted images subdirectory"
    )
    include_timestamp: bool = Field(
        default=True, description="Add timestamps to output filenames"
    )
    include_metadata: bool = Field(
        default=True, description="Include document metadata in outputs"
    )


class ProcessingConfig(BaseModel):
    """Document processing behavior and performance settings."""

    batch_size: int = Field(default=10, description="Documents per processing batch")
    max_workers: int = Field(default=4, description="Maximum worker processes")
    continue_on_error: bool = Field(
        default=True, description="Continue batch processing after individual failures"
    )
    max_retries: int = Field(
        default=2, description="Maximum retry attempts per document"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries (seconds)"
    )
    show_progress: bool = Field(default=True, description="Display progress bars")
    log_level: str = Field(default="INFO", description="Logging verbosity level")


class Config(BaseModel):
    """Main configuration object combining all settings."""

    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    def get_output_path(self, subdirectory: str) -> Path:
        """Build full path for output subdirectory."""
        return Path(self.output.output_base_dir) / subdirectory

    def ensure_output_dirs(self) -> None:
        """Create all required output directories."""
        base_dir = Path(self.output.output_base_dir)
        base_dir.mkdir(exist_ok=True)

        for subdir in [
            self.output.raw_output_dir,
            self.output.processed_output_dir,
            self.output.chunks_output_dir,
            self.output.images_output_dir,
        ]:
            (base_dir / subdir).mkdir(exist_ok=True)

    def is_supported_file(self, file_path: Path) -> bool:
        """Check if file extension is supported."""
        return file_path.suffix.lower() in self.docling.supported_formats


# Global configuration instance
config = Config()
