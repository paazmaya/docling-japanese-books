"""Basic tests for the docling-japanese-books package."""

import tempfile
from pathlib import Path

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
