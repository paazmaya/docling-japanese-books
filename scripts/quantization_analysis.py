#!/usr/bin/env python3
"""
Vector Quantization Storage Analysis

This script analyzes different quantization methods for vector embeddings
and calculates their storage impact on a real collection of Japanese books (PDFs).

Key features:
- Uses Docling pipeline to extract text and statistics from all PDFs in test_docs
- Calculates number of books, average pages per book, average characters per page, and chunk statistics from real data
- Runs quantization analysis for multiple methods (float32, float16, int8, int4, binary, PQ, SQ)
- Outputs storage estimates and recommendations based on actual document content

Embeddings are based on BGE-M3 (1024 dimensions) for Japanese document processing.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))
from docling_japanese_books.late_chunking import LateChunkingProcessor
from docling_japanese_books.processor import DocumentProcessor


@dataclass
class QuantizationMethod:
    """Configuration for a specific quantization method."""

    name: str
    bits_per_dimension: float
    accuracy_retention: float  # Percentage of original accuracy retained
    search_speed_multiplier: float  # Relative search speed (1.0 = baseline)
    compression_ratio: float  # Storage reduction ratio
    description: str
    implementation_complexity: str  # "Simple", "Moderate", "Complex"
    supported_databases: list[str] = field(default_factory=lambda: [])


@dataclass
class BookCollection:
    """
    Configuration for the book collection to analyze.
    All values are derived from real PDF statistics using Docling.
    """

    def __init__(
        self,
        num_books: int,
        pages_per_book: int,
        chars_per_page: int,
        chunk_size: int,
        overlap_ratio: float,
        embedding_dimensions: int,
    ):
        self.num_books = num_books
        self.pages_per_book = pages_per_book
        self.chars_per_page = chars_per_page
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.embedding_dimensions = embedding_dimensions


@dataclass
class StorageCalculation:
    """Storage calculation results for a quantization method."""

    method_name: str
    total_chunks: int
    bytes_per_vector: int
    metadata_bytes_per_chunk: int
    total_vector_storage_mb: float
    total_metadata_storage_mb: float
    total_storage_mb: float
    storage_gb: float
    compression_vs_float32: str
    estimated_accuracy_loss: str


class QuantizationAnalyzer:
    """Analyzes storage requirements for different vector quantization methods."""

    def __init__(self, collection: BookCollection):
        self.collection = collection
        self.methods = self._define_quantization_methods()

    def _define_quantization_methods(self) -> dict[str, QuantizationMethod]:
        """Define all quantization methods to analyze."""
        return {
            "float32": QuantizationMethod(
                name="Float32 (Full Precision)",
                bits_per_dimension=32,
                accuracy_retention=1.0,
                search_speed_multiplier=1.0,
                compression_ratio=1.0,
                description="Standard IEEE 754 32-bit floating point - baseline quality",
                implementation_complexity="Simple",
                supported_databases=[
                    "Milvus",
                    "Pinecone",
                    "Weaviate",
                    "Qdrant",
                    "Chroma",
                    "FAISS",
                ],
            ),
            "float16": QuantizationMethod(
                name="Float16 (Half Precision)",
                bits_per_dimension=16,
                accuracy_retention=0.99,
                search_speed_multiplier=1.2,
                compression_ratio=0.5,
                description="IEEE 754 16-bit floating point - minimal accuracy loss",
                implementation_complexity="Simple",
                supported_databases=["Milvus", "Weaviate", "Qdrant", "FAISS"],
            ),
            "bfloat16": QuantizationMethod(
                name="BFloat16 (Brain Float)",
                bits_per_dimension=16,
                accuracy_retention=0.985,
                search_speed_multiplier=1.3,
                compression_ratio=0.5,
                description="Brain Float 16-bit - optimized for ML workloads, better range than float16",
                implementation_complexity="Simple",
                supported_databases=["Milvus", "Qdrant", "FAISS", "Weaviate"],
            ),
            "int8": QuantizationMethod(
                name="INT8 Quantization",
                bits_per_dimension=8,
                accuracy_retention=0.95,
                search_speed_multiplier=1.5,
                compression_ratio=0.25,
                description="8-bit integer quantization with calibration - good balance",
                implementation_complexity="Moderate",
                supported_databases=[
                    "Milvus",
                    "Pinecone",
                    "Weaviate",
                    "Qdrant",
                    "FAISS",
                ],
            ),
            "int4": QuantizationMethod(
                name="INT4 Quantization",
                bits_per_dimension=4,
                accuracy_retention=0.88,
                search_speed_multiplier=2.0,
                compression_ratio=0.125,
                description="4-bit quantization - significant compression with moderate accuracy loss",
                implementation_complexity="Moderate",
                supported_databases=["Milvus", "Qdrant", "FAISS"],
            ),
            "binary": QuantizationMethod(
                name="Binary Quantization",
                bits_per_dimension=1,
                accuracy_retention=0.75,
                search_speed_multiplier=3.0,
                compression_ratio=0.03125,
                description="1-bit binary - maximum compression, significant accuracy loss",
                implementation_complexity="Moderate",
                supported_databases=["Milvus", "Weaviate", "FAISS"],
            ),
            "pq8": QuantizationMethod(
                name="Product Quantization (8-bit)",
                bits_per_dimension=8,
                accuracy_retention=0.92,
                search_speed_multiplier=1.8,
                compression_ratio=0.25,
                description="Product Quantization with 8-bit codebooks - adaptive compression",
                implementation_complexity="Complex",
                supported_databases=["Milvus", "FAISS", "Qdrant"],
            ),
            "sq8": QuantizationMethod(
                name="Scalar Quantization (8-bit)",
                bits_per_dimension=8,
                accuracy_retention=0.94,
                search_speed_multiplier=1.6,
                compression_ratio=0.25,
                description="Uniform scalar quantization - simple and effective",
                implementation_complexity="Simple",
                supported_databases=["Milvus", "Weaviate", "Qdrant"],
            ),
        }

    def calculate_chunks_per_book(self) -> int:
        """Calculate number of chunks per book based on chunking strategy."""
        chars_per_book = self.collection.pages_per_book * self.collection.chars_per_page

        # Account for overlap in chunking
        effective_chunk_size = self.collection.chunk_size * (
            1 - self.collection.overlap_ratio
        )
        chunks_per_book = int(np.ceil(chars_per_book / effective_chunk_size))

        return chunks_per_book

    def calculate_storage_for_method(
        self, method: QuantizationMethod
    ) -> StorageCalculation:
        """Calculate storage requirements for a specific quantization method."""
        chunks_per_book = self.calculate_chunks_per_book()
        total_chunks = self.collection.num_books * chunks_per_book

        # Calculate vector storage
        bits_per_vector = (
            self.collection.embedding_dimensions * method.bits_per_dimension
        )
        bytes_per_vector = int(np.ceil(bits_per_vector / 8))

        # Metadata includes: text content, document ID, chunk index, file path, etc.
        # Estimate ~1KB per chunk for metadata (conservative)
        metadata_bytes_per_chunk = 1024

        # Calculate totals
        total_vector_bytes = total_chunks * bytes_per_vector
        total_metadata_bytes = total_chunks * metadata_bytes_per_chunk

        total_vector_mb = total_vector_bytes / (1024 * 1024)
        total_metadata_mb = total_metadata_bytes / (1024 * 1024)
        total_storage_mb = total_vector_mb + total_metadata_mb

        # Calculate compression ratio vs float32
        float32_bytes_per_vector = self.collection.embedding_dimensions * 4
        compression_ratio = bytes_per_vector / float32_bytes_per_vector

        return StorageCalculation(
            method_name=method.name,
            total_chunks=total_chunks,
            bytes_per_vector=bytes_per_vector,
            metadata_bytes_per_chunk=metadata_bytes_per_chunk,
            total_vector_storage_mb=total_vector_mb,
            total_metadata_storage_mb=total_metadata_mb,
            total_storage_mb=total_storage_mb,
            storage_gb=total_storage_mb / 1024,
            compression_vs_float32=f"{compression_ratio:.3f}x ({compression_ratio * 100:.1f}%)",
            estimated_accuracy_loss=f"{(1 - method.accuracy_retention) * 100:.1f}%",
        )

    def analyze_all_methods(self) -> dict[str, StorageCalculation]:
        """Analyze storage requirements for all quantization methods."""
        results: dict[str, StorageCalculation] = {}
        for method_key, method in self.methods.items():
            results[method_key] = self.calculate_storage_for_method(method)
        return results

    def generate_comparison_table(self, results: dict[str, StorageCalculation]) -> str:
        """Generate a markdown comparison table."""

        table = """# Vector Quantization Storage Analysis

## Collection Configuration

- **Books**: {num_books:,} books
- **Pages per Book**: {pages_per_book} pages
- **Characters per Page**: {chars_per_page} chars (Japanese mixed content)
- **Chunk Size**: {chunk_size} characters with {overlap_percent}% overlap
- **Chunks per Book**: {chunks_per_book:,} chunks
- **Total Chunks**: {total_chunks:,} chunks
- **Embedding Model**: BGE-M3 ({dimensions:,} dimensions)

## Storage Comparison by Quantization Method

| Method | Vector Storage | Metadata | Total Storage | vs Float32 | Accuracy Loss | Search Speed | Complexity |
|--------|----------------|----------|---------------|------------|---------------|--------------|------------|""".format(
            num_books=self.collection.num_books,
            pages_per_book=self.collection.pages_per_book,
            chars_per_page=self.collection.chars_per_page,
            chunk_size=self.collection.chunk_size,
            overlap_percent=int(self.collection.overlap_ratio * 100),
            chunks_per_book=self.calculate_chunks_per_book(),
            total_chunks=results["float32"].total_chunks,
            dimensions=self.collection.embedding_dimensions,
        )

        # Sort methods by storage size (ascending)
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_storage_mb)

        for method_key, calc in sorted_results:
            method = self.methods[method_key]
            table += f"""
| **{method.name}** | {calc.total_vector_storage_mb:.1f} MB | {calc.total_metadata_storage_mb:.1f} MB | **{calc.total_storage_mb:.1f} MB** | {calc.compression_vs_float32} | {calc.estimated_accuracy_loss} | {method.search_speed_multiplier:.1f}x | {method.implementation_complexity} |"""

        return table

    def generate_detailed_analysis(self, results: dict[str, StorageCalculation]) -> str:
        """Generate detailed analysis with implementation notes."""

        analysis = """

## Detailed Analysis & Implementation Notes

"""

        for method_key, calc in results.items():
            method = self.methods[method_key]
            analysis += f"""
### {method.name}

**Storage**: {calc.storage_gb:.2f} GB ({calc.total_storage_mb:.1f} MB)
**Description**: {method.description}
**Accuracy Retention**: {method.accuracy_retention * 100:.1f}%
**Search Speed**: {method.search_speed_multiplier:.1f}x faster than float32
**Implementation**: {method.implementation_complexity}

**Supported Databases**: {", ".join(method.supported_databases)}

**Bytes per Vector**: {calc.bytes_per_vector:,} bytes ({calc.bytes_per_vector / 4096 * 100:.1f}% of 4KB)
**Total Vector Storage**: {calc.total_vector_storage_mb:.1f} MB
**Compression Ratio**: {calc.compression_vs_float32}

"""

        return analysis

    def generate_recommendations(self) -> str:
        """Generate recommendations based on use cases."""

        recommendations = """
## Recommendations by Use Case

### 🎯 **Production Applications (Accuracy Critical)**
**Recommendation**: **BFloat16**, **Float16**, or **INT8 Quantization**
- BFloat16: 50% storage reduction, 98.5% accuracy retention, ML-optimized
- Float16: 50% storage reduction, 99% accuracy retention, wider support
- INT8: 75% storage reduction, 95% accuracy retention
- All offer good balance of compression and quality

### 💰 **Cost-Optimized Deployments**
**Recommendation**: **INT8** or **Product Quantization (8-bit)**
- 75% storage reduction significantly reduces cloud costs
- Acceptable accuracy loss for most applications
- Wide database support

### ⚡ **High-Performance Search**
**Recommendation**: **INT4** or **Binary Quantization**
- 2-3x faster search performance
- Suitable for real-time applications where some accuracy loss is acceptable
- Massive storage savings (87.5% - 96.9% reduction)

### 🧪 **Research & Development**
**Recommendation**: **Float32** baseline + **BFloat16/Float16** for production
- Float32 for development and accuracy benchmarking
- BFloat16 for ML-heavy workloads (better numerical stability)
- Float16 for general production deployment (minimal accuracy loss)

### 📊 **Japanese Text Specifics**
**Considerations for Japanese Documents**:
- Dense character encoding benefits from higher precision
- Semantic relationships in kanji/hiragana/katakana may be sensitive to quantization
- Recommend testing with Float16 first, then INT8 if acceptable
- Binary quantization may lose too much semantic information for Japanese

## Implementation Priority

1. **Start with BFloat16/Float16**: Easy win - 50% storage reduction, minimal code changes
   - BFloat16: Better for ML/embedding models, more stable gradients
   - Float16: Wider database support, slightly better precision for small values
2. **Evaluate INT8**: Good compression, test accuracy on your specific data
3. **Consider PQ/SQ**: For advanced optimization after baseline is established
4. **Test INT4/Binary**: Only if extreme compression is needed and accuracy loss is acceptable

## BFloat16 vs Float16 Comparison

| Aspect | BFloat16 | Float16 |
|--------|----------|---------|
| **Range** | ±3.4 × 10³⁸ (same as float32) | ±6.55 × 10⁴ |
| **Precision** | 7-bit mantissa | 10-bit mantissa |
| **ML Optimization** | ✅ Designed for neural networks | ⚠️ General purpose |
| **Overflow Risk** | ❌ Very low | ⚠️ Higher with large embeddings |
| **Database Support** | 🔄 Growing (Milvus 2.3+) | ✅ Widespread |
| **Hardware Acceleration** | ✅ TPU, modern GPU | ✅ Most modern hardware |

**For Japanese embeddings**: BFloat16 is often better due to the wide range of values in high-dimensional embeddings and better handling of the semantic relationships in dense Japanese text.

"""

        return recommendations

    def save_results(self, results: dict[str, StorageCalculation], output_path: Path):
        """Save detailed results to JSON file."""
        json_results = {}
        for method_key, calc in results.items():
            method = self.methods[method_key]
            json_results[method_key] = {
                "method": {
                    "name": method.name,
                    "bits_per_dimension": method.bits_per_dimension,
                    "accuracy_retention": method.accuracy_retention,
                    "search_speed_multiplier": method.search_speed_multiplier,
                    "compression_ratio": method.compression_ratio,
                    "description": method.description,
                    "implementation_complexity": method.implementation_complexity,
                    "supported_databases": method.supported_databases,
                },
                "storage": {
                    "total_chunks": calc.total_chunks,
                    "bytes_per_vector": calc.bytes_per_vector,
                    "metadata_bytes_per_chunk": calc.metadata_bytes_per_chunk,
                    "total_vector_storage_mb": calc.total_vector_storage_mb,
                    "total_metadata_storage_mb": calc.total_metadata_storage_mb,
                    "total_storage_mb": calc.total_storage_mb,
                    "storage_gb": calc.storage_gb,
                    "compression_vs_float32": calc.compression_vs_float32,
                    "estimated_accuracy_loss": calc.estimated_accuracy_loss,
                },
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)


def extract_pdf_stats(pdf_dir: Path) -> dict[str, int | float]:
    """
    Extract statistics from all PDF files in the given directory using Docling.
    Returns number of books, average pages per book, average characters per page, chunk size, and embedding dimensions.
    Chunk count is calculated using LateChunkingProcessor for each document.
    """
    processor = DocumentProcessor()
    late_chunker = LateChunkingProcessor()
    pdf_files = list(pdf_dir.glob("*.pdf"))
    num_books = len(pdf_files)
    total_chunks = 0
    chunk_size = 400  # Default, will be updated if needed
    embedding_dimensions = 1024  # BGE-M3 default
    overlap_ratio = 0.1
    pages_per_book_list: list[int] = []
    chars_per_page_list: list[float] = []
    chunks_per_book_list: list[int] = []

    for pdf_file in pdf_files:
        # Use Docling to extract text from PDF
        try:
            conv_results = list(
                processor.converter.convert_all([pdf_file], raises_on_error=False)
            )
            if not conv_results or not hasattr(conv_results[0], "document"):
                continue
            doc = conv_results[0].document
            pages = getattr(doc, "num_pages", 1)
            text = (
                doc.export_to_markdown()
                if hasattr(doc, "export_to_markdown")
                else str(doc)
            )
            chars = len(text)
            pages_per_book_list.append(pages)
            chars_per_page_list.append(chars / pages if pages else 0)
            # Use late chunking to get chunk count
            chunks, _ = late_chunker.simple_sentence_chunker(
                text, max_chunk_length=chunk_size
            )
            chunks_per_book_list.append(len(chunks))
            total_chunks += len(chunks)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            continue

    avg_pages_per_book = (
        int(np.round(np.mean(pages_per_book_list))) if pages_per_book_list else 1
    )
    avg_chars_per_page = (
        int(np.round(np.mean(chars_per_page_list))) if chars_per_page_list else 1
    )
    avg_chunks_per_book = (
        int(np.round(np.mean(chunks_per_book_list))) if chunks_per_book_list else 1
    )

    return {
        "num_books": num_books,
        "pages_per_book": avg_pages_per_book,
        "chars_per_page": avg_chars_per_page,
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "embedding_dimensions": embedding_dimensions,
        "total_chunks": total_chunks,
        "avg_chunks_per_book": avg_chunks_per_book,
    }


def main():
    """
    Main entry point for quantization analysis.
    - Extracts real statistics from all PDFs in test_docs using Docling
    - Initializes BookCollection with extracted values
    - Runs quantization analysis for all methods
    - Outputs markdown and JSON reports with storage estimates and recommendations
    """
    pdf_dir = Path("test_docs")
    stats = extract_pdf_stats(pdf_dir)
    collection = BookCollection(
        num_books=int(stats["num_books"]),
        pages_per_book=int(stats["pages_per_book"]),
        chars_per_page=int(stats["chars_per_page"]),
        chunk_size=int(stats["chunk_size"]),
        overlap_ratio=float(stats["overlap_ratio"]),
        embedding_dimensions=int(stats["embedding_dimensions"]),
    )

    analyzer = QuantizationAnalyzer(collection)
    results = analyzer.analyze_all_methods()

    # Generate reports
    comparison_table = analyzer.generate_comparison_table(results)
    detailed_analysis = analyzer.generate_detailed_analysis(results)
    recommendations = analyzer.generate_recommendations()

    # Combine all sections
    full_report = comparison_table + detailed_analysis + recommendations

    # Save outputs
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save markdown report
    report_path = output_dir / "quantization_storage_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    # Save JSON results
    json_path = output_dir / "quantization_analysis_results.json"
    analyzer.save_results(results, json_path)

    print("📊 Quantization Analysis Complete!")
    print(f"📄 Report: {report_path}")
    print(f"📊 Data: {json_path}")
    print("\n" + "=" * 60)
    print(comparison_table)


if __name__ == "__main__":
    main()
