#!/usr/bin/env python3
"""
Evaluate Snowflake Arctic Embed L v2.0 against current BGE-M3 implementation.

This script runs a comprehensive evaluation comparing:
1. Traditional approach (all-MiniLM-L6-v2): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. BGE-M3 with Late Chunking (current implementation): https://huggingface.co/BAAI/bge-m3
3. Snowflake Arctic Embed L v2.0 with traditional chunking: https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0

The evaluation uses all PDF documents found in the test_docs/ directory.
New documents added to test_docs/ will automatically be included in future evaluations.

Usage:
    python scripts/evaluate_snowflake_arctic.py
"""

import logging
import sys
from pathlib import Path

from docling_japanese_books.embedding_evaluation import EmbeddingEvaluator

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("embedding_evaluation_snowflake.log"),
        ],
    )


def load_test_documents():
    """Load Japanese test documents from test_docs/ folder for evaluation."""
    import glob
    import os

    from docling.document_converter import DocumentConverter

    documents = {}
    test_docs_path = Path(__file__).parent.parent / "test_docs"

    # Find all PDF files in test_docs directory
    pdf_files = glob.glob(str(test_docs_path / "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {test_docs_path}")

    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(pdf_files)} PDF files from test_docs/")

    # Initialize Docling converter
    converter = DocumentConverter()

    for pdf_path in pdf_files:
        try:
            # Extract document name (without extension) for use as key
            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
            logger.info(f"Processing: {doc_name}")

            # Convert PDF to text using Docling
            result = converter.convert(pdf_path)

            # Extract main text content
            text_content = result.document.export_to_markdown()

            # Clean and truncate if necessary (keep first 3000 characters for evaluation)
            if len(text_content) > 3000:
                text_content = text_content[:3000] + "..."
                logger.info(f"Truncated {doc_name} to 3000 characters for evaluation")

            if text_content.strip():
                documents[doc_name] = text_content.strip()
                logger.info(
                    f"Successfully loaded {doc_name}: {len(text_content)} characters"
                )
            else:
                logger.warning(f"No text content extracted from {doc_name}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            continue

    if not documents:
        raise RuntimeError("No documents were successfully processed from test_docs/")

    logger.info(f"Successfully loaded {len(documents)} documents for evaluation")
    return documents


def main():
    """Run Snowflake Arctic embedding evaluation."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Snowflake Arctic Embed L v2.0 evaluation")

    # Load test documents
    documents = load_test_documents()
    logger.info(f"Loaded {len(documents)} test documents")

    # Initialize evaluator
    evaluator = EmbeddingEvaluator()

    # Run evaluation
    results_path = Path("embedding_evaluation_snowflake_results.json")
    results = evaluator.run_comparison_study(documents, results_path)

    logger.info(f"Evaluation completed. Results saved to {results_path}")

    # Additional analysis
    print("\n🎯 DETAILED MODEL COMPARISON:")
    for result in results:
        print(f"\n📄 Document: {result.document_id}")
        print(
            f"   Traditional: {result.traditional_metrics.japanese_specific_score:.3f}"
        )
        print(
            f"   BGE-M3:      {result.late_chunking_metrics.japanese_specific_score:.3f} ({result.bge_m3_improvement:+.1f}%)"
        )
        print(
            f"   Snowflake:   {result.snowflake_arctic_metrics.japanese_specific_score:.3f} ({result.snowflake_improvement:+.1f}%)"
        )
        print(f"   Winner:      {result.best_model}")


if __name__ == "__main__":
    main()
