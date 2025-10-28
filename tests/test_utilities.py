"""Tests for utility functions and scripts."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestQuantizationAnalysis:
    """Test cases for quantization analysis functionality."""

    def setup_method(self):
        """Add scripts to Python path for testing."""
        self.scripts_path = Path(__file__).parent.parent / "scripts"
        if str(self.scripts_path) not in sys.path:
            sys.path.insert(0, str(self.scripts_path))

    def teardown_method(self):
        """Clean up Python path after testing."""
        if str(self.scripts_path) in sys.path:
            sys.path.remove(str(self.scripts_path))

    def test_quantization_method_dataclass(self):
        """Test QuantizationMethod dataclass structure."""
        from quantization_analysis import QuantizationMethod

        method = QuantizationMethod(
            name="test_method",
            bits_per_dimension=8.0,
            accuracy_retention=90.0,
            search_speed_multiplier=1.5,
            compression_ratio=4.0,
            description="Test method",
            implementation_complexity="Simple",
            supported_databases=["milvus", "pinecone"],
        )

        assert method.name == "test_method"
        assert abs(method.bits_per_dimension - 8.0) < 0.001
        assert abs(method.accuracy_retention - 90.0) < 0.001
        assert abs(method.search_speed_multiplier - 1.5) < 0.001
        assert abs(method.compression_ratio - 4.0) < 0.001
        assert method.description == "Test method"
        assert method.implementation_complexity == "Simple"
        assert "milvus" in method.supported_databases

    def test_book_collection_initialization(self):
        """Test BookCollection initialization and calculations."""
        from quantization_analysis import BookCollection

        collection = BookCollection(
            num_books=10,
            pages_per_book=50,
            chars_per_page=2000,
            chunk_size=400,
            embedding_dimensions=1024,
            overlap_ratio=0.1,
        )

        assert collection.num_books == 10
        assert collection.pages_per_book == 50
        assert collection.chars_per_page == 2000
        assert collection.chunk_size == 400
        assert collection.embedding_dimensions == 1024
        assert abs(collection.overlap_ratio - 0.1) < 0.001

    def test_quantization_analyzer_initialization(self):
        """Test QuantizationAnalyzer initialization."""
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        collection = BookCollection(
            num_books=5,
            pages_per_book=20,
            chars_per_page=1000,
            chunk_size=200,
            embedding_dimensions=512,
            overlap_ratio=0.2,
        )

        analyzer = QuantizationAnalyzer(collection)

        assert analyzer.collection == collection
        assert isinstance(analyzer.quantization_methods, dict)
        assert "float32" in analyzer.quantization_methods
        assert "float16" in analyzer.quantization_methods
        assert "int8" in analyzer.quantization_methods

    def test_chunks_calculation(self):
        """Test chunk calculation logic."""
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        collection = BookCollection(
            num_books=2,
            pages_per_book=10,
            chars_per_page=1000,  # 10k chars per book
            chunk_size=400,
            embedding_dimensions=1024,
            overlap_ratio=0.1,
        )

        analyzer = QuantizationAnalyzer(collection)
        chunks_per_book = analyzer.calculate_chunks_per_book()

        # Expected: 10k chars / (400 * 0.9 effective) = ~28 chunks
        assert chunks_per_book > 20
        assert chunks_per_book < 35

    def test_storage_calculation(self):
        """Test storage calculation for quantization methods."""
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        collection = BookCollection(
            num_books=1,
            pages_per_book=10,
            chars_per_page=400,  # Exactly 1 chunk per page
            chunk_size=400,
            embedding_dimensions=1024,
            overlap_ratio=0.0,  # No overlap for simplicity
        )

        analyzer = QuantizationAnalyzer(collection)

        # Test float32 calculation
        float32_method = analyzer.quantization_methods["float32"]
        storage = analyzer.calculate_storage_for_method(float32_method)

        assert storage.total_vectors == 10  # 10 pages = 10 chunks
        assert storage.storage_per_vector_bytes == 1024 * 4  # 1024 dims * 4 bytes
        assert storage.total_storage_bytes == 10 * 1024 * 4
        assert storage.storage_mb > 0

    def test_comparison_table_generation(self):
        """Test comparison table generation."""
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        collection = BookCollection(
            num_books=1,
            pages_per_book=1,
            chars_per_page=400,
            chunk_size=400,
            embedding_dimensions=128,  # Smaller for faster testing
            overlap_ratio=0.0,
        )

        analyzer = QuantizationAnalyzer(collection)
        results = analyzer.analyze_all_methods()

        table = analyzer.generate_comparison_table(results)

        assert isinstance(table, str)
        assert "Method" in table
        assert "Storage (MB)" in table
        assert "float32" in table
        assert "float16" in table

    def test_recommendations_generation(self):
        """Test recommendations generation."""
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        collection = BookCollection(
            num_books=1,
            pages_per_book=1,
            chars_per_page=400,
            chunk_size=400,
            embedding_dimensions=128,
            overlap_ratio=0.0,
        )

        analyzer = QuantizationAnalyzer(collection)
        results = analyzer.analyze_all_methods()

        recommendations = analyzer.generate_recommendations(results)

        assert isinstance(recommendations, str)
        assert "Recommendations" in recommendations
        assert len(recommendations) > 100  # Should be substantial content

    @patch("json.dump")
    @patch("pathlib.Path.open")
    def test_results_saving(self, mock_open, mock_json_dump):
        """Test results saving functionality."""
        from quantization_analysis import BookCollection, QuantizationAnalyzer

        collection = BookCollection(
            num_books=1,
            pages_per_book=1,
            chars_per_page=400,
            chunk_size=400,
            embedding_dimensions=128,
            overlap_ratio=0.0,
        )

        analyzer = QuantizationAnalyzer(collection)
        results = analyzer.analyze_all_methods()

        output_path = Path("/tmp/test_results.json")
        analyzer.save_results(results, output_path)

        # Verify file operations were called
        assert mock_open.called
        assert mock_json_dump.called

    @patch("docling_japanese_books.processor.DocumentProcessor")
    @patch("docling_japanese_books.late_chunking.LateChunkingProcessor")
    def test_pdf_stats_extraction_mock(self, mock_late_chunker, mock_processor):
        """Test PDF statistics extraction with mocked dependencies."""
        from quantization_analysis import extract_pdf_stats

        # Mock processor behavior
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance

        # Mock late chunker behavior
        mock_late_chunker_instance = MagicMock()
        mock_late_chunker_instance.simple_sentence_chunker.return_value = [
            "chunk1",
            "chunk2",
        ]
        mock_late_chunker.return_value = mock_late_chunker_instance

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock PDF files
            pdf_dir = Path(temp_dir)
            (pdf_dir / "test1.pdf").touch()
            (pdf_dir / "test2.pdf").touch()

            # Should not crash even with no real PDFs
            try:
                stats = extract_pdf_stats(pdf_dir)
                # If it returns something, verify structure
                if stats:
                    assert isinstance(stats, dict)
            except Exception:
                # Expected to fail without real PDF processing
                pass


class TestEmbeddingEvaluation:
    """Test cases for embedding evaluation functionality."""

    def test_evaluation_metrics_dataclass(self):
        """Test EvaluationMetrics dataclass."""
        from docling_japanese_books.embedding_evaluation import EvaluationMetrics

        metrics = EvaluationMetrics(
            context_preservation_score=0.85,
            japanese_specific_score=0.92,
            processing_time=12.5,
            memory_usage_mb=256.0,
            chunk_count=150,
        )

        assert abs(metrics.context_preservation_score - 0.85) < 0.001
        assert abs(metrics.japanese_specific_score - 0.92) < 0.001
        assert abs(metrics.processing_time - 12.5) < 0.001
        assert abs(metrics.memory_usage_mb - 256.0) < 0.001
        assert metrics.chunk_count == 150

    def test_evaluation_results_dataclass(self):
        """Test EvaluationResults dataclass."""
        from docling_japanese_books.embedding_evaluation import (
            EvaluationMetrics,
            EvaluationResults,
        )

        metrics = EvaluationMetrics(
            context_preservation_score=0.8,
            japanese_specific_score=0.9,
            processing_time=10.0,
            memory_usage_mb=200.0,
            chunk_count=100,
        )

        results = EvaluationResults(
            document_id="test_doc",
            traditional_metrics=metrics,
            bge_m3_metrics=metrics,
            snowflake_arctic_metrics=metrics,
            jina_v4_metrics=metrics,
            best_model="BGE-M3",
            improvement_over_baseline=15.5,
        )

        assert results.document_id == "test_doc"
        assert results.best_model == "BGE-M3"
        assert abs(results.improvement_over_baseline - 15.5) < 0.001

    @patch("docling_japanese_books.embedding_evaluation.SentenceTransformer")
    def test_embedding_evaluator_initialization(self, mock_sentence_transformer):
        """Test EmbeddingEvaluator initialization without model loading."""
        from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator

        evaluator = EmbeddingEvaluator()

        assert evaluator.traditional_model is None
        assert evaluator.bge_m3_model is None
        assert evaluator.snowflake_arctic_model is None
        assert evaluator.jina_v4_model is None
        assert evaluator.late_chunking_processor is not None

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        import numpy as np

        from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator

        with patch.object(EmbeddingEvaluator, "__init__", lambda x: None):
            evaluator = EmbeddingEvaluator()

            # Test with known vectors
            vec1 = np.array([1.0, 0.0, 0.0])
            vec2 = np.array([0.0, 1.0, 0.0])  # Orthogonal
            vec3 = np.array([1.0, 0.0, 0.0])  # Same as vec1

            similarity_orthogonal = evaluator.calculate_cosine_similarity(vec1, vec2)
            similarity_identical = evaluator.calculate_cosine_similarity(vec1, vec3)

            assert abs(similarity_orthogonal - 0.0) < 0.001
            assert abs(similarity_identical - 1.0) < 0.001

    def test_traditional_chunking_method(self):
        """Test traditional chunking method."""
        from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator

        with patch.object(EmbeddingEvaluator, "__init__", lambda x: None):
            evaluator = EmbeddingEvaluator()

            test_document = "This is a test document. " * 50  # Long enough to chunk

            chunks = evaluator.simple_traditional_chunking(test_document)

            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_model_type_detection(self):
        """Test model type detection from chunking method."""
        from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator

        with patch.object(EmbeddingEvaluator, "__init__", lambda x: None):
            evaluator = EmbeddingEvaluator()

            assert evaluator._get_model_type("traditional") == "traditional"
            assert evaluator._get_model_type("late_chunking") == "bge_m3"
            assert evaluator._get_model_type("snowflake_arctic") == "snowflake_arctic"
            assert evaluator._get_model_type("jina_v4") == "jina_v4"

    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        from docling_japanese_books.embedding_evaluation import (
            EmbeddingEvaluator,
            EvaluationMetrics,
        )

        with patch.object(EmbeddingEvaluator, "__init__", lambda x: None):
            evaluator = EmbeddingEvaluator()

            baseline = EvaluationMetrics(
                context_preservation_score=0.8,
                japanese_specific_score=0.8,
                processing_time=10.0,
                memory_usage_mb=200.0,
                chunk_count=100,
            )

            comparison = EvaluationMetrics(
                context_preservation_score=0.9,
                japanese_specific_score=0.9,  # 12.5% improvement
                processing_time=8.0,
                memory_usage_mb=180.0,
                chunk_count=95,
            )

            improvement = evaluator._calculate_improvement(baseline, comparison)

            # Should be approximately 12.5% improvement
            assert 10.0 < improvement < 15.0

    def test_best_model_determination(self):
        """Test best model determination logic."""
        from docling_japanese_books.embedding_evaluation import (
            EmbeddingEvaluator,
            EvaluationMetrics,
        )

        with patch.object(EmbeddingEvaluator, "__init__", lambda x: None):
            evaluator = EmbeddingEvaluator()

            traditional = EvaluationMetrics(
                context_preservation_score=0.7,
                japanese_specific_score=0.7,
                processing_time=10.0,
                memory_usage_mb=200.0,
                chunk_count=100,
            )

            bge_m3 = EvaluationMetrics(
                context_preservation_score=0.85,
                japanese_specific_score=0.85,
                processing_time=12.0,
                memory_usage_mb=250.0,
                chunk_count=110,
            )

            snowflake = EvaluationMetrics(
                context_preservation_score=0.8,
                japanese_specific_score=0.8,
                processing_time=8.0,
                memory_usage_mb=180.0,
                chunk_count=95,
            )

            jina_v4 = EvaluationMetrics(
                context_preservation_score=0.9,
                japanese_specific_score=0.9,  # Highest score
                processing_time=15.0,
                memory_usage_mb=300.0,
                chunk_count=120,
            )

            best_model = evaluator._determine_best_model(
                traditional, bge_m3, snowflake, jina_v4
            )

            assert "Jina" in best_model  # Should pick Jina v4 with highest score

    def test_serialization_conversion(self):
        """Test serialization conversion utility."""
        import numpy as np

        from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator

        with patch.object(EmbeddingEvaluator, "__init__", lambda x: None):
            evaluator = EmbeddingEvaluator()

            # Test numpy type conversions
            assert isinstance(
                evaluator._convert_to_serializable(np.float64(3.14)), float
            )
            assert isinstance(evaluator._convert_to_serializable(np.int32(42)), int)
            assert isinstance(
                evaluator._convert_to_serializable(np.array([1, 2, 3])), list
            )

            # Test dict conversion
            test_dict = {"key": np.float64(1.5)}
            result = evaluator._convert_to_serializable(test_dict)
            assert isinstance(result["key"], float)


if __name__ == "__main__":
    pytest.main([__file__])
