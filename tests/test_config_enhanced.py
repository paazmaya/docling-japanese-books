"""Enhanced configuration tests to improve coverage."""

import tempfile
from pathlib import Path

import pytest

from docling_japanese_books.config import DatabaseConfig, config


def test_database_config_cloud_error_conditions():
    """Test error conditions in cloud database configuration."""
    # Test missing cloud URI
    db_config = DatabaseConfig(deployment_mode="cloud")

    with pytest.raises(ValueError, match="Zilliz Cloud URI is required"):
        db_config.get_connection_uri()

    # Test cloud mode configuration without error (API key validation happens at connection time)
    db_config_with_uri = DatabaseConfig(
        deployment_mode="cloud", zilliz_cloud_uri="https://test.cloud.zilliz.com"
    )

    params = db_config_with_uri.get_connection_params()
    assert params["uri"] == "https://test.cloud.zilliz.com"


def test_database_config_cloud_success():
    """Test successful cloud database configuration."""
    db_config = DatabaseConfig(
        deployment_mode="cloud",
        zilliz_cloud_uri="https://test.cloud.zilliz.com",
        zilliz_api_key="test_key_12345",
    )

    # Test URI retrieval
    uri = db_config.get_connection_uri()
    assert uri == "https://test.cloud.zilliz.com"

    # Test connection parameters with token
    params = db_config.get_connection_params()
    assert params["uri"] == "https://test.cloud.zilliz.com"
    assert params["token"] == "test_key_12345"


def test_database_config_local_mode():
    """Test local database configuration."""
    db_config = DatabaseConfig(deployment_mode="local", milvus_uri=".database/test.db")

    # Test URI retrieval
    uri = db_config.get_connection_uri()
    assert uri == ".database/test.db"

    # Test connection parameters without token
    params = db_config.get_connection_params()
    assert params["uri"] == ".database/test.db"
    assert "token" not in params


def test_database_config_docker_mode():
    """Test docker database configuration."""
    db_config = DatabaseConfig(
        deployment_mode="docker", milvus_uri="http://localhost:19530"
    )

    # Test URI retrieval
    uri = db_config.get_connection_uri()
    assert uri == "http://localhost:19530"

    # Test connection parameters without token
    params = db_config.get_connection_params()
    assert params["uri"] == "http://localhost:19530"
    assert "token" not in params


def test_config_output_path_edge_cases():
    """Test edge cases in output path generation."""
    # Test empty subdirectory
    empty_path = config.get_output_path("")
    assert isinstance(empty_path, Path)

    # Test subdirectory with spaces
    space_path = config.get_output_path("test with spaces")
    assert isinstance(space_path, Path)
    assert "test with spaces" in str(space_path)

    # Test subdirectory with special characters
    special_path = config.get_output_path("test-data_123")
    assert isinstance(special_path, Path)
    assert "test-data_123" in str(special_path)


def test_config_directory_creation_edge_cases():
    """Test edge cases in directory creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_base = config.output.output_base_dir

        try:
            # Test with existing directory first
            temp_path = Path(temp_dir) / "test_output"
            temp_path.mkdir(parents=True, exist_ok=True)
            config.output.output_base_dir = str(temp_path)

            # Should create subdirectories
            config.ensure_output_dirs()

            # Verify structure
            assert temp_path.exists()
            assert (temp_path / "raw").exists()
            assert (temp_path / "processed").exists()
            assert (temp_path / "chunks").exists()
            assert (temp_path / "images").exists()

        finally:
            config.output.output_base_dir = original_base


def test_file_support_comprehensive():
    """Test comprehensive file support detection."""
    # Test all supported formats explicitly
    supported_formats = [
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
    ]

    for fmt in supported_formats:
        test_file = Path(f"test{fmt}")
        assert config.is_supported_file(test_file) is True, (
            f"Format {fmt} should be supported"
        )

    # Test case insensitivity
    assert config.is_supported_file(Path("test.PDF")) is True
    assert config.is_supported_file(Path("test.DOCX")) is True
    assert config.is_supported_file(Path("test.JPG")) is True

    # Test unsupported formats
    unsupported_formats = [
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".mp3",
        ".zip",
        ".exe",
        ".dmg",
        ".iso",
        ".tar",
        ".gz",
        ".rar",
    ]

    for fmt in unsupported_formats:
        test_file = Path(f"test{fmt}")
        assert config.is_supported_file(test_file) is False, (
            f"Format {fmt} should not be supported"
        )


def test_config_field_ranges_comprehensive():
    """Test comprehensive configuration field ranges."""
    # Test chunking configuration bounds
    assert 1 <= config.chunking.max_chunk_tokens <= 100000
    assert 0 <= config.chunking.chunk_overlap <= config.chunking.max_chunk_tokens
    assert 1 <= config.chunking.min_chunk_length <= config.chunking.max_chunk_tokens

    # Test docling configuration bounds
    assert 0.1 <= config.docling.images_scale <= 10.0
    assert 1 <= config.docling.max_file_size_mb <= 10000
    assert 1 <= config.docling.max_num_pages <= 1000000
    assert 1 <= config.docling.thread_count <= 100

    # Test processing configuration bounds
    assert 1 <= config.processing.batch_size <= 10000
    assert 1 <= config.processing.max_workers <= 100
    assert config.processing.max_retries >= 0
    assert config.processing.retry_delay >= 0.0

    # Test database configuration bounds
    assert 1 <= config.database.embedding_dimension <= 100000
    assert config.database.connection_timeout > 0


def test_config_string_validations():
    """Test configuration string field validations."""
    # Test model names follow HuggingFace convention
    assert "/" in config.chunking.tokenizer_model
    assert len(config.chunking.tokenizer_model.split("/")) == 2

    assert "/" in config.chunking.embedding_model
    assert len(config.chunking.embedding_model.split("/")) == 2

    assert "/" in config.docling.vision_model_repo_id
    assert len(config.docling.vision_model_repo_id.split("/")) >= 2

    # Test collection name is valid
    assert config.database.collection_name
    assert len(config.database.collection_name) > 0
    assert config.database.collection_name.replace("_", "").replace("-", "").isalnum()

    # Test chunking strategy is valid
    assert config.chunking.chunking_strategy in [
        "auto",
        "late",
        "traditional",
        "hybrid",
        "hierarchical",
    ]

    # Test database type is valid
    assert config.database.database_type in ["milvus", "pinecone", "chroma", "weaviate"]

    # Test deployment mode is valid
    assert config.database.deployment_mode in ["local", "docker", "cloud"]


def test_config_output_formats():
    """Test output format configuration."""
    # Test that output formats are valid
    valid_formats = {"json", "jsonl", "markdown", "txt", "csv", "parquet"}

    for fmt in config.output.output_formats:
        assert fmt in valid_formats, f"Unknown output format: {fmt}"

    # Test that at least one format is specified
    assert len(config.output.output_formats) > 0

    # Test that required formats are present
    assert "json" in config.output.output_formats
    assert "markdown" in config.output.output_formats


def test_config_path_separators():
    """Test path configurations handle different separators correctly."""
    # Test forward slash paths
    forward_path = config.get_output_path("sub/directory")
    assert isinstance(forward_path, Path)

    # Test that paths are normalized
    normalized_path = config.get_output_path("sub//double//slash")
    assert "//" not in str(normalized_path)

    # Test relative path components
    relative_path = config.get_output_path("../parent")
    assert isinstance(relative_path, Path)


def test_config_boolean_combinations():
    """Test boolean configuration combinations."""
    # Test vision-related booleans are consistent
    if config.docling.enable_vision:
        assert config.docling.vision_model
        assert config.docling.vision_model_repo_id

    # Test output-related booleans
    assert isinstance(config.output.include_timestamp, bool)
    assert isinstance(config.output.include_metadata, bool)

    # Test processing-related booleans
    assert isinstance(config.processing.continue_on_error, bool)
    assert isinstance(config.processing.show_progress, bool)


def test_config_consistency_checks():
    """Test internal configuration consistency."""
    # Test chunk overlap is reasonable percentage of chunk size
    overlap_percentage = (
        config.chunking.chunk_overlap / config.chunking.max_chunk_tokens
    )
    assert 0.0 <= overlap_percentage <= 0.5, (
        "Chunk overlap should be at most 50% of chunk size"
    )

    # Test min chunk length is reasonable
    assert config.chunking.min_chunk_length <= config.chunking.max_chunk_tokens
    assert (
        config.chunking.min_chunk_length <= config.chunking.chunk_overlap
        or config.chunking.chunk_overlap == 0
    )

    # Test thread count is reasonable for the system
    assert config.docling.thread_count <= 64  # Reasonable upper bound

    # Test batch size is reasonable
    assert config.processing.batch_size <= 1000  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
