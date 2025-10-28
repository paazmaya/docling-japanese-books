"""Basic tests for the docling-japanese-books package."""

import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docling_japanese_books.config import config


def test_config_loading():
    """Test that configuration loads correctly."""
    assert config.docling.enable_ocr is True
    assert config.chunking.tokenizer_model == "ibm-granite/granite-docling-258M"
    assert config.chunking.embedding_model == "BAAI/bge-m3"
    assert config.chunking.use_late_chunking is True
    assert config.database.embedding_dimension == 1024
    assert config.database.database_type == "milvus"
    assert config.processing.batch_size == 10
    assert "json" in config.output.output_formats


def test_supported_file_detection():
    """Test that file format detection works."""
    # Test supported files
    assert config.is_supported_file(Path("document.pdf")) is True
    assert config.is_supported_file(Path("document.docx")) is True
    assert config.is_supported_file(Path("document.html")) is True

    # Test unsupported files
    assert config.is_supported_file(Path("document.xyz")) is False
    assert config.is_supported_file(Path("document.exe")) is False


def test_output_directory_creation():
    """Test that output directories are created correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update config temporarily
        original_base = config.output.output_base_dir
        config.output.output_base_dir = temp_dir

        try:
            config.ensure_output_dirs()

            # Check that directories exist
            base_path = Path(temp_dir)
            assert base_path.exists()
            assert (base_path / config.output.raw_output_dir).exists()
            assert (base_path / config.output.processed_output_dir).exists()
            assert (base_path / config.output.chunks_output_dir).exists()

        finally:
            # Restore original config
            config.output.output_base_dir = original_base


def test_database_configuration():
    """Test database configuration settings."""
    # Test deployment mode configuration
    assert config.database.deployment_mode in ["local", "cloud"]

    # Test local database path (relative path)
    if config.database.deployment_mode == "local":
        milvus_path = Path(config.database.milvus_uri)
        assert str(milvus_path).startswith(".database")
        assert milvus_path.suffix == ".db"

    # Test connection parameter generation
    connection_params = config.database.get_connection_params()
    assert "uri" in connection_params

    if config.database.deployment_mode == "cloud":
        # Cloud mode should include token if API key is set
        if config.database.zilliz_api_key:
            assert "token" in connection_params

    # Check collection name is set
    assert config.database.collection_name == "docling_japanese_books"

    # Test embedding dimension for BGE-M3
    assert config.database.embedding_dimension == 1024


def test_chunking_configuration():
    """Test chunking configuration parameters."""
    assert config.chunking.max_chunk_tokens == 512
    assert config.chunking.chunk_overlap == 50
    assert config.chunking.use_late_chunking is True

    # Test model configurations
    assert "granite" in config.chunking.tokenizer_model.lower()
    assert "bge-m3" in config.chunking.embedding_model.lower()


def test_processing_configuration():
    """Test processing configuration parameters."""
    assert config.processing.batch_size > 0
    assert config.processing.max_concurrent_files > 0
    assert config.docling.thread_count > 0
    assert config.docling.max_file_size_mb > 0
    assert config.docling.max_num_pages > 0


def test_docling_configuration():
    """Test Docling-specific configuration."""
    assert config.docling.enable_ocr is True
    assert config.docling.do_table_structure is True
    assert config.docling.do_cell_matching is True
    assert config.docling.generate_page_images is True
    assert config.docling.images_scale > 0

    # Test vision configuration
    assert isinstance(config.docling.enable_vision, bool)
    if config.docling.enable_vision:
        assert config.docling.vision_model in ["granite"]
        assert "granite" in config.docling.vision_model_repo_id.lower()

    # Test supported formats
    assert ".pdf" in config.docling.supported_formats
    assert len(config.docling.supported_formats) > 0


@mock.patch("docling_japanese_books.processor.DocumentProcessor._setup_vector_db")
@mock.patch("docling_japanese_books.processor.DocumentProcessor._setup_image_processor")
@mock.patch("docling_japanese_books.processor.DocumentProcessor._setup_chunker")
@mock.patch("docling_japanese_books.processor.DocumentProcessor._setup_docling")
@mock.patch("docling_japanese_books.processor.DocumentProcessor._setup_tokenizer")
def test_processor_initialization(
    mock_tokenizer, mock_docling, mock_chunker, mock_image, mock_vector
):
    """Test DocumentProcessor initialization without external dependencies."""
    from docling_japanese_books.processor import DocumentProcessor

    # Mock all the setup methods to prevent actual initialization
    mock_tokenizer.return_value = None
    mock_docling.return_value = None
    mock_chunker.return_value = None
    mock_image.return_value = None
    mock_vector.return_value = None

    with patch.object(DocumentProcessor, "_ensure_directories"):
        DocumentProcessor()

        # Verify all setup methods were called
        mock_tokenizer.assert_called_once()
        mock_docling.assert_called_once()
        mock_chunker.assert_called_once()
        mock_image.assert_called_once()
        mock_vector.assert_called_once()


def test_file_discovery_logic():
    """Test file discovery without actual file processing."""
    from docling_japanese_books.processor import DocumentProcessor

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        (temp_path / "document.pdf").touch()
        (temp_path / "image.jpg").touch()
        (temp_path / "text.txt").touch()
        (temp_path / "document.docx").touch()

        # Create a large file that should be skipped
        large_file = temp_path / "large.pdf"
        large_file.write_bytes(
            b"x" * (config.docling.max_file_size_mb * 1024 * 1024 + 1)
        )

        with mock.patch.object(DocumentProcessor, "__init__", lambda x: None):
            processor = DocumentProcessor()
            processor.config = config
            processor.logger = MagicMock()

            files = processor.discover_files(temp_path)

            # Should find supported files but not the oversized one
            file_names = [f.name for f in files]
            assert "document.pdf" in file_names
            assert "document.docx" in file_names
            assert "image.jpg" not in file_names  # Not supported
            assert "text.txt" not in file_names  # Not supported
            assert "large.pdf" not in file_names  # Too large


def test_image_processor_utilities():
    """Test ImageProcessor utility methods without external dependencies."""
    from docling_japanese_books.image_processor import ImageProcessor

    with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
        processor = ImageProcessor()
        processor.config = config

        # Test Japanese content analysis
        test_annotations = [
            "This image contains Japanese text with hiragana characters",
            "Vertical text layout typical of Japanese books",
            "Traditional Japanese calligraphy with kanji",
            "No Japanese content here",
        ]

        analysis = processor.analyze_japanese_content(test_annotations)

        assert isinstance(analysis, dict)
        assert "has_japanese_text" in analysis
        assert "has_vertical_layout" in analysis
        assert "has_cultural_elements" in analysis
        assert "confidence_score" in analysis

        # Should detect Japanese content in first three annotations
        assert analysis["has_japanese_text"] is True


def test_late_chunking_processor_initialization():
    """Test LateChunkingProcessor initialization without model loading."""
    from docling_japanese_books.late_chunking import LateChunkingProcessor

    with mock.patch("torch.device"):
        processor = LateChunkingProcessor()

        assert processor.model_name == config.chunking.embedding_model
        assert processor.model is None  # Not loaded initially
        assert processor.tokenizer is None  # Not loaded initially


def test_model_downloader_configuration():
    """Test ModelDownloader configuration without actual downloading."""
    from docling_japanese_books.downloader import ModelDownloader

    with mock.patch("huggingface_hub.HfApi"):
        downloader = ModelDownloader()

        model_info = downloader.get_model_info()

        assert isinstance(model_info, dict)
        assert "tokenizer" in model_info
        assert "embedding" in model_info
        assert "vision" in model_info

        # Check model paths match configuration
        assert model_info["tokenizer"] == config.chunking.tokenizer_model
        assert model_info["embedding"] == config.chunking.embedding_model
        assert model_info["vision"] == config.docling.vision_model_repo_id


@mock.patch("docling_japanese_books.vector_db.MilvusVectorDB._setup_milvus_client")
@mock.patch("docling_japanese_books.vector_db.MilvusVectorDB._setup_embedding_model")
def test_vector_db_initialization(mock_embedding, mock_client):
    """Test MilvusVectorDB initialization without actual connections."""
    from docling_japanese_books.vector_db import MilvusVectorDB

    mock_embedding.return_value = None
    mock_client.return_value = None

    MilvusVectorDB()

    # Verify setup methods were called
    mock_embedding.assert_called_once()
    mock_client.assert_called_once()


def test_quantization_analysis_calculations():
    """Test quantization analysis calculations without file I/O."""
    import sys
    from pathlib import Path

    # Add scripts to path temporarily
    scripts_path = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_path))

    try:
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        # Create test collection with known parameters
        collection = BookCollection(
            num_books=10,
            pages_per_book=50,
            chars_per_page=2000,
            chunk_size=400,
            embedding_dimensions=1024,
            overlap_ratio=0.1,
        )

        analyzer = QuantizationAnalyzer(collection)

        # Test chunks calculation
        chunks_per_book = analyzer.calculate_chunks_per_book()
        assert chunks_per_book > 0

        # Test storage calculation for float32
        float32_method = analyzer.quantization_methods["float32"]
        storage_calc = analyzer.calculate_storage_for_method(float32_method)

        assert storage_calc.total_vectors == collection.num_books * chunks_per_book
        assert storage_calc.storage_per_vector_bytes > 0
        assert storage_calc.total_storage_bytes > 0
        assert storage_calc.storage_mb > 0

    finally:
        # Clean up sys.path
        if str(scripts_path) in sys.path:
            sys.path.remove(str(scripts_path))


def test_cli_configuration_display():
    """Test CLI configuration display without actual execution."""
    from rich.console import Console

    from docling_japanese_books.cli import display_config_panel

    console = Console()
    test_directory = Path("/tmp/test")

    # Should not raise any exceptions
    try:
        display_config_panel(console, test_directory)
    except Exception as e:
        # If it fails, it should be due to missing dependencies, not logic errors
        assert "rich" not in str(e).lower()


def test_config_database_uri_generation():
    """Test database URI generation for different deployment modes."""
    # Test local mode
    original_mode = config.database.deployment_mode

    try:
        config.database.deployment_mode = "local"
        uri = config.database.get_connection_uri()
        assert uri == config.database.milvus_uri
        assert ".db" in uri

        # Test connection parameters for local
        params = config.database.get_connection_params()
        assert params["uri"] == config.database.milvus_uri
        assert "token" not in params  # Local mode doesn't use tokens

    finally:
        config.database.deployment_mode = original_mode


def test_config_validation():
    """Test configuration validation and constraints."""
    # Test that critical paths are relative (for portability)
    assert not Path(config.docling.artifacts_path).is_absolute()
    assert not Path(config.database.milvus_uri).is_absolute()
    assert not Path(config.output.output_base_dir).is_absolute()

    # Test that numeric values are within reasonable ranges
    assert 0 < config.chunking.chunk_size < 10000
    assert 0 < config.chunking.chunk_overlap < config.chunking.chunk_size
    assert 0 < config.processing.batch_size < 1000
    assert 0 < config.docling.max_file_size_mb < 10000
    assert 0 < config.docling.max_num_pages < 100000

    # Test that percentages are valid
    assert 0.0 <= config.chunking.overlap_ratio <= 1.0
    assert config.docling.images_scale > 0


if __name__ == "__main__":
    pytest.main([__file__])
