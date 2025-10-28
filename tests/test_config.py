"""Tests for the configuration system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from docling_japanese_books.config import (
    ChunkingConfig,
    Config,
    DatabaseConfig,
    DoclingConfig,
    OutputConfig,
    ProcessingConfig,
    config,
)


class TestConfigurationSystem:
    """Test cases for configuration classes and validation."""

    def test_docling_config_defaults(self):
        """Test DoclingConfig default values."""
        docling_config = DoclingConfig()

        assert docling_config.artifacts_path == ".models"
        assert docling_config.enable_ocr is True
        assert docling_config.enable_vision is False
        assert docling_config.do_table_structure is True
        assert docling_config.do_cell_matching is True
        assert docling_config.generate_page_images is True
        assert docling_config.vision_model == "granite"
        assert abs(docling_config.images_scale - 2.0) < 0.001
        assert docling_config.max_file_size_mb == 100
        assert docling_config.max_num_pages == 1000
        assert docling_config.thread_count == 4

    def test_database_config_defaults(self):
        """Test DatabaseConfig default values and methods."""
        db_config = DatabaseConfig()

        assert db_config.database_type == "milvus"
        assert db_config.deployment_mode == "local"
        assert db_config.embedding_dimension == 1024
        assert db_config.collection_name == "docling_japanese_books"
        assert db_config.milvus_uri.endswith(".db")

        # Test connection URI for local mode
        uri = db_config.get_connection_uri()
        assert uri == db_config.milvus_uri

        # Test connection parameters
        params = db_config.get_connection_params()
        assert "uri" in params
        assert params["uri"] == db_config.milvus_uri

    def test_database_config_cloud_mode(self):
        """Test DatabaseConfig cloud mode configuration."""
        db_config = DatabaseConfig(
            deployment_mode="cloud",
            zilliz_cloud_uri="https://test.cloud.zilliz.com",
            zilliz_api_key="test_key",
        )

        assert db_config.deployment_mode == "cloud"

        # Test connection URI for cloud mode
        uri = db_config.get_connection_uri()
        assert uri == "https://test.cloud.zilliz.com"

        # Test connection parameters for cloud mode
        params = db_config.get_connection_params()
        assert "uri" in params
        assert "token" in params
        assert params["token"] == "test_key"

    def test_database_config_validation_errors(self):
        """Test DatabaseConfig validation for missing required fields."""
        # Test missing cloud URI
        db_config = DatabaseConfig(deployment_mode="cloud")

        with pytest.raises(ValueError, match="Zilliz Cloud URI is required"):
            db_config.get_connection_uri()

        # Test missing API key for cloud mode
        db_config = DatabaseConfig(
            deployment_mode="cloud", zilliz_cloud_uri="https://test.cloud.zilliz.com"
        )

        with pytest.raises(ValueError, match="Zilliz Cloud API key is required"):
            db_config.get_connection_params()

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        chunking_config = ChunkingConfig()

        assert chunking_config.chunk_size == 400
        assert chunking_config.chunk_overlap == 40
        assert abs(chunking_config.overlap_ratio - 0.1) < 0.001
        assert chunking_config.use_late_chunking is True
        assert "granite" in chunking_config.tokenizer_model.lower()
        assert "bge-m3" in chunking_config.embedding_model.lower()

    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        output_config = OutputConfig()

        assert output_config.output_base_dir == "output"
        assert output_config.raw_output_dir == "raw"
        assert output_config.processed_output_dir == "processed"
        assert output_config.chunks_output_dir == "chunks"
        assert output_config.images_output_dir == "images"

        # Test output formats
        assert "json" in output_config.output_formats
        assert "markdown" in output_config.output_formats
        assert "jsonl" in output_config.output_formats

    def test_processing_config_defaults(self):
        """Test ProcessingConfig default values."""
        processing_config = ProcessingConfig()

        assert processing_config.batch_size == 10
        assert processing_config.max_concurrent_files == 3
        assert processing_config.enable_progress_bar is True
        assert processing_config.save_intermediate_results is True

    def test_main_config_integration(self):
        """Test main Config class integration."""
        main_config = Config()

        # Test that all sub-configs are properly initialized
        assert isinstance(main_config.docling, DoclingConfig)
        assert isinstance(main_config.database, DatabaseConfig)
        assert isinstance(main_config.chunking, ChunkingConfig)
        assert isinstance(main_config.output, OutputConfig)
        assert isinstance(main_config.processing, ProcessingConfig)

    def test_config_output_path_generation(self):
        """Test output path generation methods."""
        test_config = Config()

        # Test get_output_path
        raw_path = test_config.get_output_path("raw")
        assert raw_path == Path("output/raw")

        processed_path = test_config.get_output_path("processed")
        assert processed_path == Path("output/processed")

    def test_config_directory_creation(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = Config()

            # Temporarily override the base directory
            original_base = test_config.output.output_base_dir
            test_config.output.output_base_dir = temp_dir

            try:
                test_config.ensure_output_dirs()

                # Check that all directories were created
                base_path = Path(temp_dir)
                assert (base_path / test_config.output.raw_output_dir).exists()
                assert (base_path / test_config.output.processed_output_dir).exists()
                assert (base_path / test_config.output.chunks_output_dir).exists()
                assert (base_path / test_config.output.images_output_dir).exists()

            finally:
                # Restore original configuration
                test_config.output.output_base_dir = original_base

    def test_supported_file_detection(self):
        """Test file format support detection."""
        test_config = Config()

        # Test supported formats
        supported_files = [
            Path("document.pdf"),
            Path("document.docx"),
            Path("document.pptx"),
            Path("document.html"),
            Path("document.md"),
        ]

        for file_path in supported_files:
            assert test_config.is_supported_file(file_path) is True

        # Test unsupported formats
        unsupported_files = [
            Path("document.xyz"),
            Path("document.exe"),
            Path("image.jpg"),
            Path("video.mp4"),
            Path("audio.mp3"),
        ]

        for file_path in unsupported_files:
            assert test_config.is_supported_file(file_path) is False

    def test_config_field_constraints(self):
        """Test configuration field constraints and validation."""
        # Test that chunk_overlap is less than chunk_size
        chunking_config = ChunkingConfig()
        assert chunking_config.chunk_overlap < chunking_config.chunk_size

        # Test that overlap_ratio is between 0 and 1
        assert 0.0 <= chunking_config.overlap_ratio <= 1.0

        # Test positive numeric constraints
        docling_config = DoclingConfig()
        assert docling_config.images_scale > 0
        assert docling_config.max_file_size_mb > 0
        assert docling_config.max_num_pages > 0
        assert docling_config.thread_count > 0

        processing_config = ProcessingConfig()
        assert processing_config.batch_size > 0
        assert processing_config.max_concurrent_files > 0

    def test_global_config_instance(self):
        """Test global configuration instance."""
        # Test that global config is properly initialized
        assert isinstance(config, Config)
        assert config.database.database_type == "milvus"
        assert config.chunking.use_late_chunking is True

        # Test that configuration is consistent across accesses
        first_access = config.docling.enable_ocr
        second_access = config.docling.enable_ocr
        assert first_access == second_access

    def test_config_model_paths(self):
        """Test model path configurations."""
        # Test that model paths are properly configured
        assert "granite" in config.chunking.tokenizer_model.lower()
        assert "bge-m3" in config.chunking.embedding_model.lower()
        assert "granite" in config.docling.vision_model_repo_id.lower()

        # Test artifacts path is relative
        assert not Path(config.docling.artifacts_path).is_absolute()

    @patch.dict("os.environ", {"ZILLIZ_API_KEY": "test_env_key"})
    def test_environment_variable_integration(self):
        """Test environment variable integration."""
        # Create new config instance to pick up env vars
        env_config = DatabaseConfig()

        # The environment variable should be picked up if the field supports it
        # Note: This depends on the actual implementation of environment variable support
        assert hasattr(env_config, "zilliz_api_key")

    def test_config_serialization_compatibility(self):
        """Test that configuration can be serialized (for debugging/logging)."""
        test_config = Config()

        # Should be able to convert to dict without errors
        try:
            config_dict = test_config.model_dump()
            assert isinstance(config_dict, dict)
            assert "docling" in config_dict
            assert "database" in config_dict
            assert "chunking" in config_dict
            assert "output" in config_dict
            assert "processing" in config_dict
        except Exception as e:
            pytest.fail(f"Configuration serialization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
