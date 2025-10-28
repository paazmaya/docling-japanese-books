"""Minimal tests that can run without external model dependencies."""

import tempfile
from pathlib import Path

import pytest

from docling_japanese_books.config import config


def test_config_basic_loading():
    """Test that basic configuration loads without errors."""
    assert config is not None
    assert hasattr(config, "docling")
    assert hasattr(config, "database")
    assert hasattr(config, "chunking")
    assert hasattr(config, "output")
    assert hasattr(config, "processing")


def test_config_values_reasonable():
    """Test that configuration values are reasonable."""
    # Test basic types and ranges
    assert isinstance(config.docling.enable_ocr, bool)
    assert isinstance(config.docling.max_file_size_mb, int)
    assert config.docling.max_file_size_mb > 0

    assert isinstance(config.chunking.max_chunk_tokens, int)
    assert config.chunking.max_chunk_tokens > 0
    assert config.chunking.chunk_overlap < config.chunking.max_chunk_tokens

    assert isinstance(config.database.embedding_dimension, int)
    assert config.database.embedding_dimension > 0

    assert isinstance(config.processing.batch_size, int)
    assert config.processing.batch_size > 0


def test_supported_file_types():
    """Test file type support detection."""
    # Supported types (based on actual config)
    assert config.is_supported_file(Path("test.pdf")) is True
    assert config.is_supported_file(Path("document.docx")) is True
    assert config.is_supported_file(Path("image.jpg")) is True  # Images are supported
    assert (
        config.is_supported_file(Path("document.txt")) is True
    )  # Text files are supported

    # Unsupported types
    assert config.is_supported_file(Path("video.mp4")) is False
    assert config.is_supported_file(Path("unknown.xyz")) is False
    assert config.is_supported_file(Path("audio.wav")) is False


def test_output_directory_management():
    """Test output directory creation and path management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Backup original setting
        original_base = config.output.output_base_dir

        try:
            # Set temporary directory
            config.output.output_base_dir = temp_dir

            # Test path generation
            raw_path = config.get_output_path("raw")
            assert isinstance(raw_path, Path)
            assert str(raw_path).endswith("raw")

            # Test directory creation
            config.ensure_output_dirs()

            # Verify directories exist
            base_path = Path(temp_dir)
            assert (base_path / "raw").exists()
            assert (base_path / "processed").exists()
            assert (base_path / "chunks").exists()
            assert (base_path / "images").exists()

        finally:
            # Restore original setting
            config.output.output_base_dir = original_base


def test_database_configuration():
    """Test database configuration without actual connection."""
    # Test URI generation
    uri = config.database.get_connection_uri()
    assert isinstance(uri, str)
    assert len(uri) > 0

    # Test connection parameters
    params = config.database.get_connection_params()
    assert isinstance(params, dict)
    assert "uri" in params

    # Test configuration values
    assert config.database.collection_name
    assert config.database.embedding_dimension > 0


def test_model_configuration_strings():
    """Test that model configuration strings are valid."""
    # Test model names are non-empty strings
    assert isinstance(config.chunking.tokenizer_model, str)
    assert len(config.chunking.tokenizer_model) > 0
    assert "/" in config.chunking.tokenizer_model  # Should be HF format

    assert isinstance(config.chunking.embedding_model, str)
    assert len(config.chunking.embedding_model) > 0
    assert "/" in config.chunking.embedding_model  # Should be HF format

    assert isinstance(config.docling.vision_model_repo_id, str)
    assert len(config.docling.vision_model_repo_id) > 0


def test_path_configurations():
    """Test that path configurations are reasonable."""
    # Test that local paths are relative (for portability)
    assert not Path(config.docling.artifacts_path).is_absolute()
    assert not Path(config.output.output_base_dir).is_absolute()

    # Test that database URI is valid
    db_uri = config.database.milvus_uri
    assert isinstance(db_uri, str)
    assert len(db_uri) > 0
    # URI can be either local file path (.db) or network URL (http/https)
    assert db_uri.endswith(".db") or db_uri.startswith(("http://", "https://"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
