"""Basic tests for the docling-japanese-books package."""

import tempfile
from pathlib import Path

from docling_japanese_books.config import config


def test_config_loading():
    """Test that configuration loads correctly."""
    assert config.docling.enable_ocr is True
    assert config.chunking.tokenizer_model == "ibm-granite/granite-docling-258M"
    assert config.chunking.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
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


def test_milvus_database_path():
    """Test that Milvus database path is in user home directory."""
    milvus_path = Path(config.database.milvus_uri)
    home_dir = Path.home()

    # Check that path starts with home directory
    assert str(milvus_path).startswith(str(home_dir))

    # Check that it's in .milvus subdirectory
    assert ".milvus" in milvus_path.parts

    # Check collection name is set
    assert config.database.collection_name == "docling_japanese_books"
