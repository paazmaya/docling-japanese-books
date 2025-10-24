"""Configuration settings for the document processing pipeline with cloud support."""

import os
from pathlib import Path

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
        default=".database/docling_documents.db",
        description="Local Milvus Lite database path",
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
        default_factory=lambda: os.getenv("MILVUS_DEPLOYMENT_MODE", "local"),
        description="Database deployment mode: 'local' or 'cloud'",
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

    def get_connection_params(self) -> dict:
        """Build Milvus client connection parameters."""
        params = {
            "uri": self.get_connection_uri(),
        }

        if self.deployment_mode == "cloud":
            if not self.zilliz_api_key:
                raise ValueError(
                    "Zilliz Cloud API key is required for cloud deployment mode"
                )
            params["token"] = self.zilliz_api_key

        return params


class ChunkingConfig(BaseModel):
    """Text chunking and embedding configuration for Japanese documents."""

    tokenizer_model: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="Granite Docling model for document tokenization",
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="BGE-M3 multilingual embedding model",
    )
    max_chunk_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(default=50, description="Token overlap between chunks")
    min_chunk_length: int = Field(
        default=20, description="Minimum chunk length (tokens)"
    )
    chunking_strategy: str = Field(
        default="hybrid", description="Chunking strategy type"
    )
    merge_list_items: bool = Field(
        default=True, description="Merge list items during chunking"
    )
    use_late_chunking: bool = Field(
        default=True,
        description="Enable Late Chunking for better context preservation",
    )
    merge_peers: bool = Field(
        default=True, description="Merge peer chunks in hybrid strategy"
    )


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
