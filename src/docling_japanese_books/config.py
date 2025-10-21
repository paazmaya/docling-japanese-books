"""Hardcoded configuration settings for the document processing pipeline."""

from pathlib import Path

# Built-in typing support for Python 3.9+
from pydantic import BaseModel, Field


class DoclingConfig(BaseModel):
    """Docling processing configuration with hardcoded settings."""

    # Model settings - hardcoded for consistency
    artifacts_path: str = Field(
        default=".models",
        description="Path to store Docling models in project directory",
    )

    # PDF Processing options
    enable_ocr: bool = Field(default=True, description="Enable OCR processing")
    enable_vision: bool = Field(
        default=True, description="Enable vision model processing for image description"
    )
    do_table_structure: bool = Field(
        default=True, description="Enable table structure extraction"
    )
    do_cell_matching: bool = Field(
        default=True, description="Enable PDF cell matching for tables"
    )
    generate_page_images: bool = Field(
        default=True,
        description="Generate page images for HTML output and separate storage",
    )

    # Vision model settings for Japanese books
    vision_model: str = Field(
        default="granite",
        description="Vision model to use: granite, smolvlm, or custom",
    )
    # https://huggingface.co/ibm-granite/granite-vision-3.3-2b
    vision_model_repo_id: str = Field(
        default="ibm-granite/granite-vision-3.3-2b",
        description="HuggingFace model repository ID for vision model",
    )
    vision_prompt: str = Field(
        default="Describe this image in detail, paying special attention to any Japanese text, characters, or cultural elements. Include information about layout, visual elements, and any text content visible in the image.",
        description="Prompt for vision model when describing images",
    )
    images_scale: float = Field(
        default=2.0, description="Scale factor for image processing quality"
    )

    # Resource limits
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    max_num_pages: int = Field(default=1000, description="Maximum pages per document")
    thread_count: int = Field(default=4, description="Number of processing threads")

    # Supported formats - hardcoded list
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
        description="Supported file formats",
    )


class DatabaseConfig(BaseModel):
    """Database configuration with hardcoded settings for Milvus."""

    # Milvus configuration - using standard shared location
    database_type: str = Field(default="milvus", description="Database type")

    # Milvus Lite database path - stored in project directory
    milvus_uri: str = Field(
        default=".database/docling_documents.db",
        description="Milvus Lite database URI in project directory",
    )

    # Collection settings
    collection_name: str = Field(
        default="docling_japanese_books", description="Milvus collection name"
    )

    # Vector settings
    embedding_dimension: int = Field(
        default=384,
        description="Embedding dimension for sentence-transformers/all-MiniLM-L6-v2",
    )

    # Connection settings
    connection_timeout: int = Field(
        default=30, description="Database connection timeout"
    )
    consistency_level: str = Field(
        default="Strong",
        description="Milvus consistency level (Strong, Session, Bounded, Eventually)",
    )


class ChunkingConfig(BaseModel):
    """Chunking configuration with hardcoded settings."""

    # IBM Granite Docling model optimized for document processing
    # This model is specifically designed for document understanding tasks
    # and provides better tokenization for structured documents
    # https://huggingface.co/ibm-granite/granite-docling-258M
    tokenizer_model: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="IBM Granite Docling model for document-aware tokenization",
    )

    # Embedding model for vector storage (separate from tokenizer)
    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for generating embeddings",
    )

    # Chunk settings
    max_chunk_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(default=50, description="Token overlap between chunks")
    min_chunk_length: int = Field(
        default=20, description="Minimum chunk length in tokens"
    )

    # Chunking strategy
    chunking_strategy: str = Field(
        default="hybrid", description="Chunking strategy (hybrid/hierarchical)"
    )
    merge_list_items: bool = Field(
        default=True, description="Merge list items in hierarchical chunking"
    )
    merge_peers: bool = Field(
        default=True, description="Merge peer chunks in hybrid chunking"
    )


class OutputConfig(BaseModel):
    """Output configuration with hardcoded settings."""

    # Output formats - hardcoded for LLM training
    output_formats: list[str] = Field(
        default=["json", "markdown", "jsonl"],
        description="Output formats to generate",
    )

    # Output directories
    output_base_dir: str = Field(
        default="./output", description="Base output directory"
    )
    raw_output_dir: str = Field(default="raw", description="Raw output subdirectory")
    processed_output_dir: str = Field(
        default="processed", description="Processed output subdirectory"
    )
    chunks_output_dir: str = Field(
        default="chunks", description="Chunks output subdirectory"
    )
    images_output_dir: str = Field(
        default="images", description="Images output subdirectory for separate storage"
    )

    # File naming
    include_timestamp: bool = Field(
        default=True, description="Include timestamp in output filenames"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in output"
    )


class ProcessingConfig(BaseModel):
    """Processing configuration with hardcoded settings."""

    # Batch processing
    batch_size: int = Field(default=10, description="Documents per batch")
    max_workers: int = Field(default=4, description="Maximum worker processes")

    # Error handling
    continue_on_error: bool = Field(
        default=True, description="Continue processing on individual document errors"
    )
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")

    # Progress reporting
    show_progress: bool = Field(default=True, description="Show progress bars")
    log_level: str = Field(default="INFO", description="Logging level")


class Config(BaseModel):
    """Main configuration combining all settings."""

    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    def get_output_path(self, subdirectory: str) -> Path:
        """Get the full output path for a subdirectory."""
        return Path(self.output.output_base_dir) / subdirectory

    def ensure_output_dirs(self) -> None:
        """Ensure all output directories exist."""
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
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.docling.supported_formats


# Global configuration instance
config = Config()
