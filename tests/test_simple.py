"""Simple, reliable unit tests that work without external dependencies."""

import tempfile
from pathlib import Path

import pytest


def test_basic_imports():
    """Test that all basic modules can be imported."""
    # Test configuration import
    from docling_japanese_books.config import config

    assert config is not None

    # Test that config has expected structure
    assert hasattr(config, "docling")
    assert hasattr(config, "database")
    assert hasattr(config, "chunking")
    assert hasattr(config, "output")
    assert hasattr(config, "processing")


def test_configuration_basic_values():
    """Test basic configuration values are reasonable."""
    from docling_japanese_books.config import config

    # Database config
    assert isinstance(config.database.database_type, str)
    assert config.database.database_type == "milvus"
    assert isinstance(config.database.embedding_dimension, int)
    assert config.database.embedding_dimension > 0

    # Chunking config
    assert isinstance(config.chunking.max_chunk_tokens, int)
    assert config.chunking.max_chunk_tokens > 0
    assert isinstance(config.chunking.chunk_overlap, int)
    assert config.chunking.chunk_overlap >= 0
    assert config.chunking.chunk_overlap < config.chunking.max_chunk_tokens

    # Processing config
    assert isinstance(config.processing.batch_size, int)
    assert config.processing.batch_size > 0


def test_file_type_support():
    """Test file type support detection works."""
    from docling_japanese_books.config import config

    # Test that basic functionality works
    pdf_supported = config.is_supported_file(Path("test.pdf"))
    docx_supported = config.is_supported_file(Path("test.docx"))
    unknown_supported = config.is_supported_file(Path("test.unknown"))

    # Basic checks - at least PDF should be supported
    assert isinstance(pdf_supported, bool)
    assert isinstance(docx_supported, bool)
    assert isinstance(unknown_supported, bool)
    assert pdf_supported is True  # PDF should definitely be supported


def test_output_path_generation():
    """Test output path generation works."""
    from docling_japanese_books.config import config

    # Test path generation
    raw_path = config.get_output_path("raw")
    processed_path = config.get_output_path("processed")

    assert isinstance(raw_path, Path)
    assert isinstance(processed_path, Path)
    assert "raw" in str(raw_path)
    assert "processed" in str(processed_path)


def test_directory_creation():
    """Test directory creation functionality."""
    from docling_japanese_books.config import config

    with tempfile.TemporaryDirectory() as temp_dir:
        # Backup and modify config temporarily
        original_base = config.output.output_base_dir
        config.output.output_base_dir = temp_dir

        try:
            # Test directory creation
            config.ensure_output_dirs()

            # Check directories exist
            base_path = Path(temp_dir)
            assert base_path.exists()

            # Check expected subdirectories
            raw_dir = base_path / "raw"
            processed_dir = base_path / "processed"
            chunks_dir = base_path / "chunks"
            images_dir = base_path / "images"

            assert raw_dir.exists()
            assert processed_dir.exists()
            assert chunks_dir.exists()
            assert images_dir.exists()

        finally:
            # Restore original config
            config.output.output_base_dir = original_base


def test_database_connection_params():
    """Test database connection parameter generation."""
    from docling_japanese_books.config import config

    # Test connection URI generation
    uri = config.database.get_connection_uri()
    assert isinstance(uri, str)
    assert len(uri) > 0

    # Test connection parameters
    params = config.database.get_connection_params()
    assert isinstance(params, dict)
    assert "uri" in params
    assert params["uri"] == uri


def test_model_configuration():
    """Test model configuration strings."""
    from docling_japanese_books.config import config

    # Test tokenizer model
    tokenizer_model = config.chunking.tokenizer_model
    assert isinstance(tokenizer_model, str)
    assert len(tokenizer_model) > 0
    assert "/" in tokenizer_model  # Should be HuggingFace format

    # Test embedding model
    embedding_model = config.chunking.embedding_model
    assert isinstance(embedding_model, str)
    assert len(embedding_model) > 0
    assert "/" in embedding_model  # Should be HuggingFace format


def test_path_safety():
    """Test that configured paths are safe and relative."""
    from docling_japanese_books.config import config

    # Test artifacts path is relative
    artifacts_path = Path(config.docling.artifacts_path)
    assert not artifacts_path.is_absolute()

    # Test output path is relative
    output_path = Path(config.output.output_base_dir)
    # Allow both relative and explicitly relative (./output)
    assert not output_path.is_absolute()


def test_numerical_constraints():
    """Test that numerical configuration values are within reasonable ranges."""
    from docling_japanese_books.config import config

    # Test chunk size constraints
    assert 1 <= config.chunking.max_chunk_tokens <= 10000
    assert 0 <= config.chunking.chunk_overlap < config.chunking.max_chunk_tokens
    assert config.chunking.min_chunk_length > 0

    # Test processing constraints
    assert 1 <= config.processing.batch_size <= 1000

    # Test docling constraints
    assert config.docling.max_file_size_mb > 0
    assert config.docling.max_num_pages > 0
    assert config.docling.thread_count > 0
    assert config.docling.images_scale > 0

    # Test database constraints
    assert config.database.embedding_dimension > 0


def test_boolean_flags():
    """Test boolean configuration flags."""
    from docling_japanese_books.config import config

    # Test docling boolean flags
    assert isinstance(config.docling.enable_ocr, bool)
    assert isinstance(config.docling.enable_vision, bool)
    assert isinstance(config.docling.do_table_structure, bool)
    assert isinstance(config.docling.do_cell_matching, bool)
    assert isinstance(config.docling.generate_page_images, bool)

    # Test chunking boolean flags
    assert isinstance(config.chunking.use_late_chunking, bool)
    assert isinstance(config.chunking.merge_list_items, bool)
    assert isinstance(config.chunking.merge_peers, bool)

    # Test output boolean flags
    assert isinstance(config.output.include_timestamp, bool)
    assert isinstance(config.output.include_metadata, bool)


def test_supported_formats_list():
    """Test that supported formats list is reasonable."""
    from docling_japanese_books.config import config

    formats = config.docling.supported_formats
    assert isinstance(formats, list)
    assert len(formats) > 0

    # Should contain at least PDF
    assert ".pdf" in formats

    # All formats should start with dot
    for fmt in formats:
        assert isinstance(fmt, str)
        assert fmt.startswith(".")
        assert len(fmt) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
