#!/usr/bin/env python3
"""Simple test runner for basic functionality without heavy dependencies."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that basic imports work."""
    try:
        from docling_japanese_books.config import config

        print("✓ Configuration import successful")

        # Test basic config access
        print(f"✓ Database type: {config.database.database_type}")
        print(f"✓ Max chunk tokens: {config.chunking.max_chunk_tokens}")
        print(f"✓ Embedding model: {config.chunking.embedding_model}")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_file_detection():
    """Test file detection logic."""
    try:
        from docling_japanese_books.config import config

        # Test supported files
        pdf_result = config.is_supported_file(Path("test.pdf"))
        docx_result = config.is_supported_file(Path("doc.docx"))
        print(f"✓ PDF support: {pdf_result}")
        print(f"✓ DOCX support: {docx_result}")

        # Test unsupported files
        jpg_result = config.is_supported_file(Path("image.jpg"))
        mp4_result = config.is_supported_file(Path("video.mp4"))
        print(f"✓ JPG support (should be False): {jpg_result}")
        print(f"✓ MP4 support (should be False): {mp4_result}")

        # Basic assertion that at least PDF is supported
        if pdf_result:
            print("✓ File detection works correctly")
            return True
        else:
            print("✗ PDF files should be supported")
            return False

    except Exception as e:
        print(f"✗ File detection failed: {e}")
        return False


def test_path_management():
    """Test path management without creating files."""
    try:
        from docling_japanese_books.config import config

        # Test path generation
        raw_path = config.get_output_path("raw")
        processed_path = config.get_output_path("processed")

        print(f"✓ Raw path: {raw_path}")
        print(f"✓ Processed path: {processed_path}")

        # Test that paths are reasonable
        assert "raw" in str(raw_path)
        assert "processed" in str(processed_path)

        return True
    except Exception as e:
        print(f"✗ Path management failed: {e}")
        return False


def test_database_config():
    """Test database configuration."""
    try:
        from docling_japanese_books.config import config

        # Test URI generation
        uri = config.database.get_connection_uri()
        params = config.database.get_connection_params()

        print(f"✓ Database URI: {uri}")
        print(f"✓ Connection params keys: {list(params.keys())}")

        assert isinstance(uri, str)
        assert isinstance(params, dict)
        assert "uri" in params

        return True
    except Exception as e:
        print(f"✗ Database config failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running basic functionality tests...\n")

    tests = [
        ("Basic Imports", test_imports),
        ("File Detection", test_file_detection),
        ("Path Management", test_path_management),
        ("Database Config", test_database_config),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Running {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED\n")
            else:
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}\n")

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
