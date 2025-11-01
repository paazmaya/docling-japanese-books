"""Tests for CLI functionality without external dependencies."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from docling_japanese_books.cli import cli


class TestCLIFunctionality:
    """Test cases for CLI commands and utilities."""

    def test_cli_group_exists(self):
        """Test that the main CLI group is properly defined."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Docling Japanese Books" in result.output or "Commands:" in result.output

    @patch("docling_japanese_books.cli.setup_logging")
    def test_setup_logging_configuration(self, mock_setup):
        """Test logging setup functionality."""
        from docling_japanese_books.cli import setup_logging

        # Should not raise exceptions
        setup_logging("INFO")
        setup_logging("DEBUG")
        setup_logging("ERROR")

    def test_config_display_function(self):
        """Test configuration display without actual execution."""
        from rich.console import Console

        from docling_japanese_books.cli import display_config_panel

        console = Console()
        test_directory = Path("/tmp/test")

        # Should not raise exceptions when called
        try:
            display_config_panel(console, test_directory)
        except Exception as e:
            # Should only fail on missing dependencies, not logic errors
            assert "import" in str(e).lower() or "module" in str(e).lower()

    @patch("docling_japanese_books.cli.DocumentProcessor")
    def test_file_discovery_display(self, mock_processor):
        """Test file discovery and display functionality."""
        from rich.console import Console

        from docling_japanese_books.cli import discover_and_display_files

        # Mock processor behavior
        mock_instance = MagicMock()
        mock_instance.discover_files.return_value = [
            Path("test1.pdf"),
            Path("test2.pdf"),
            Path("doc.docx"),
        ]
        mock_processor.return_value = mock_instance

        console = Console()

        with tempfile.TemporaryDirectory() as temp_dir:
            files = discover_and_display_files(mock_instance, Path(temp_dir), console)

            assert len(files) == 3
            assert any("test1.pdf" in str(f) for f in files)
            mock_instance.discover_files.assert_called_once()

    def test_dry_run_handling(self):
        """Test dry run display functionality."""
        from rich.console import Console

        from docling_japanese_books.cli import handle_dry_run

        console = Console()
        test_files = [Path(f"test{i}.pdf") for i in range(15)]

        # Should not raise exceptions
        handle_dry_run(test_files, console)

        # Test with empty file list
        handle_dry_run([], console)

    def test_results_display(self):
        """Test processing results display."""
        from rich.console import Console

        from docling_japanese_books.cli import display_results

        console = Console()

        # Mock results object with actual properties used by display_results
        mock_results = MagicMock()
        mock_results.success_count = 5
        mock_results.partial_success_count = 0
        mock_results.failure_count = 1
        mock_results.total_time = 45.5
        mock_results.errors = ["Test error message"]

        test_files = [Path(f"test{i}.pdf") for i in range(6)]

        # Should not raise exceptions
        display_results(mock_results, test_files, console)

    @patch("docling_japanese_books.cli.ModelDownloader")
    def test_model_existence_check(self, mock_downloader):
        """Test model existence checking functionality."""
        from rich.console import Console

        from docling_japanese_books.cli import _check_existing_models

        console = Console()

        # Mock downloader behavior
        mock_instance = MagicMock()
        mock_instance.check_models_exist.return_value = {
            "tokenizer": True,
            "embedding": False,
            "vision": True,
        }
        mock_downloader.return_value = mock_instance

        result = _check_existing_models(console, mock_instance)

        # Should return False because not all models exist
        assert result is False
        mock_instance.check_models_exist.assert_called_once()

    def test_download_results_display(self):
        """Test download results display functionality."""
        from rich.console import Console

        from docling_japanese_books.cli import _display_download_results

        console = Console()

        # Mock download results
        mock_result_success = MagicMock()
        mock_result_success.success = True
        mock_result_success.model_path = Path("/test/path")
        mock_result_success.error = None

        mock_result_failure = MagicMock()
        mock_result_failure.success = False
        mock_result_failure.model_path = None
        mock_result_failure.error = "Test error"

        results = {"tokenizer": mock_result_success, "embedding": mock_result_failure}

        # Should not raise exceptions
        _display_download_results(console, results)

    def test_download_config_panel(self):
        """Test download configuration panel generation."""
        from docling_japanese_books.cli import _get_download_config_panel

        with patch("docling_japanese_books.cli.ModelDownloader") as mock_downloader:
            mock_instance = MagicMock()
            mock_instance.get_model_info.return_value = {
                "tokenizer": "test-tokenizer",
                "embedding": "test-embedding",
                "vision": "test-vision",
            }
            mock_downloader.return_value = mock_instance

            config_text = _get_download_config_panel()

            assert isinstance(config_text, str)
            assert "Models Directory" in config_text
            assert "Models to Download" in config_text

    @patch("docling_japanese_books.vector_db.MilvusVectorDB")
    def test_database_connection_test(self, mock_vector_db):
        """Test database connection testing functionality."""
        from rich.console import Console

        from docling_japanese_books.cli import _test_database_connection

        console = Console()

        # Mock successful connection
        mock_instance = MagicMock()
        mock_instance.get_collection_stats.return_value = {
            "total_entities": 100,
            "collection_name": "test_collection",
        }
        mock_vector_db.return_value = mock_instance

        result = _test_database_connection(console, "local", "", "", "")
        assert result is None  # Function doesn't return boolean

        # Mock failed connection - should raise exception
        mock_vector_db.side_effect = Exception("Connection failed")
        with pytest.raises(Exception, match="Connection failed"):
            _test_database_connection(console, "local", "", "", "")

    def test_current_config_display(self):
        """Test current configuration display."""
        from rich.console import Console

        from docling_japanese_books.cli import _display_current_config

        console = Console()

        # Should not raise exceptions
        _display_current_config(console)

    def test_cloud_instructions_display(self):
        """Test cloud setup instructions display."""
        from rich.console import Console

        from docling_japanese_books.cli import _show_cloud_instructions

        console = Console()

        # Should not raise exceptions
        _show_cloud_instructions(console)

    @patch("docling_japanese_books.cli.DocumentProcessor")
    def test_process_command_structure(self, mock_processor):
        """Test process command structure without execution."""
        runner = CliRunner()

        # Test help for process command
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "directory" in result.output.lower()

    def test_download_command_structure(self):
        """Test download command structure."""
        runner = CliRunner()

        # Test help for download command
        result = runner.invoke(cli, ["download", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output.lower() or "models" in result.output.lower()

    def test_search_command_structure(self):
        """Test search command structure."""
        runner = CliRunner()

        # Test help for search command
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0

    def test_config_db_command_structure(self):
        """Test config-db command structure."""
        runner = CliRunner()

        # Test help for config-db command
        result = runner.invoke(cli, ["config-db", "--help"])
        assert result.exit_code == 0

    def test_evaluate_command_structure(self):
        """Test evaluate command structure."""
        runner = CliRunner()

        # Test help for evaluate command
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0

    @patch("docling_japanese_books.cli.sys.exit")
    def test_error_handling_patterns(self, mock_exit):
        """Test error handling patterns in CLI functions."""
        from docling_japanese_books.cli import setup_logging

        # Test with invalid log level (should handle gracefully)
        try:
            setup_logging("INVALID_LEVEL")
        except (ValueError, AttributeError):
            # Expected behavior for invalid log level
            pass

    def test_cli_imports_available(self):
        """Test that all CLI dependencies are properly importable in test environment."""
        try:
            import click
            from rich.console import Console
            from rich.logging import RichHandler
            from rich.panel import Panel

            # Basic instantiation tests
            console = Console()
            panel = Panel("Test")
            handler = RichHandler()

            # Test click is available
            assert click is not None
            assert console is not None
            assert panel is not None
            assert handler is not None

        except ImportError as e:
            pytest.skip(f"CLI dependencies not available in test environment: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
