"""Tests for the image processing functionality."""

import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest

from docling_japanese_books.image_processor import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    def test_japanese_content_analysis(self):
        """Test Japanese content analysis without external dependencies."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            # Test with Japanese indicators
            japanese_annotations = [
                "This image contains Japanese text with hiragana characters",
                "Vertical text layout typical of Japanese books",
                "Traditional Japanese calligraphy with kanji 漢字",
                "カタカナ text visible in the image",
            ]

            analysis = processor.analyze_japanese_content(japanese_annotations)

            assert isinstance(analysis, dict)
            assert analysis["has_japanese_text"] is True
            assert analysis["confidence_score"] > 0

            # Test with non-Japanese content
            english_annotations = [
                "This is a regular English document",
                "No special characters or layouts here",
                "Standard horizontal text layout",
            ]

            analysis = processor.analyze_japanese_content(english_annotations)
            assert analysis["has_japanese_text"] is False

    def test_image_references_generation(self):
        """Test image reference text generation."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            # Test with mock image data
            test_images = [
                {
                    "hash": "abc123",
                    "filename": "image1.png",
                    "caption": "Test image caption",
                    "annotations": ["Japanese text visible", "Traditional layout"],
                },
                {
                    "hash": "def456",
                    "filename": "image2.png",
                    "caption": "Another test image",
                    "annotations": [],
                },
            ]

            references = processor.get_image_references_for_text(test_images)

            assert isinstance(references, str)
            assert "abc123" in references
            assert "def456" in references
            assert "image1.png" in references
            assert "image2.png" in references

            # Test with empty input
            empty_references = processor.get_image_references_for_text([])
            assert empty_references == ""

    def test_japanese_indicators_detection(self):
        """Test Japanese writing system indicator detection."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            # Test Japanese indicators
            test_cases = [
                ("Text with hiragana characters", True),
                ("Contains katakana カタカナ", True),
                ("Kanji characters 漢字 present", True),
                ("Japanese writing system detected", True),
                ("Regular English text only", False),
                ("Numbers 12345 and symbols !@#", False),
            ]

            for text, should_detect in test_cases:
                analysis = {}
                processor._check_japanese_indicators(text, analysis)

                if should_detect:
                    assert analysis.get("japanese_writing_systems", False)
                # Note: Method only sets indicators when found, doesn't set False

    def test_layout_orientation_detection(self):
        """Test layout and orientation indicator detection."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            test_cases = [
                ("Vertical text layout detected", True),
                ("Traditional horizontal arrangement", True),
                ("縦書き style text", True),
                ("Right-to-left reading order", True),
                ("Normal paragraph text", False),
                ("Standard formatting", False),
            ]

            for text, should_detect in test_cases:
                analysis = {}
                processor._check_layout_orientation(text, analysis)

                if should_detect:
                    assert analysis.get("layout_cues", False)

    def test_cultural_elements_detection(self):
        """Test Japanese cultural elements detection."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            test_cases = [
                ("Traditional Japanese architecture visible", True),
                ("Shrine or temple structure", True),
                ("Cherry blossoms sakura in image", True),
                ("Japanese garden elements", True),
                ("Modern office building", False),
                ("Standard landscape photo", False),
            ]

            for text, should_detect in test_cases:
                analysis = {}
                processor._check_cultural_elements(text, analysis)

                if should_detect:
                    assert analysis.get("cultural_elements", False)

    def test_timestamp_generation(self):
        """Test timestamp generation utility."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            timestamp = processor._get_timestamp()

            assert isinstance(timestamp, str)
            assert len(timestamp) > 10  # Should be ISO format
            assert "T" in timestamp  # ISO format separator

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.write_text")
    def test_image_manifest_creation(self, mock_write, mock_mkdir):
        """Test image manifest file creation without actual file I/O."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()
            processor.logger = MagicMock()
            processor.config = MagicMock()
            processor.config.get_output_path.return_value = MagicMock()

            test_images = [
                {"hash": "abc123", "filename": "test1.png"},
                {"hash": "def456", "filename": "test2.png"},
            ]

            # Should not raise exceptions
            processor.create_image_manifest("test_doc", test_images)

            # Verify file operations were attempted
            assert mock_mkdir.called or mock_write.called

    def test_legacy_image_references(self):
        """Test legacy image reference format generation."""
        with mock.patch.object(ImageProcessor, "__init__", lambda x: None):
            processor = ImageProcessor()

            test_images = [
                {
                    "hash": "abc123",
                    "filename": "test1.png",
                    "annotations": ["Test annotation 1", "Test annotation 2"],
                }
            ]

            references = processor.get_legacy_image_references_for_text(test_images)

            assert isinstance(references, str)
            if test_images:  # Should generate content for non-empty input
                assert len(references) > 0

            # Test with empty input
            empty_refs = processor.get_legacy_image_references_for_text([])
            assert empty_refs == ""


if __name__ == "__main__":
    pytest.main([__file__])
