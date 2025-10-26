"""Document processor using Docling with hardcoded configurations."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import AutoTokenizer

from .config import config
from .image_processor import ImageProcessor
from .vector_db import MilvusVectorDB


@dataclass
class ProcessingResults:
    """Results from document processing."""

    success_count: int = 0
    partial_success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)


class DocumentProcessor:
    """Main document processor using Docling."""

    def __init__(self) -> None:
        """Initialize processor with preconfigured settings for Japanese documents."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self._setup_tokenizer()
        self._setup_docling()
        self._setup_chunker()
        self._setup_vector_db()
        self._setup_image_processor()
        self._ensure_directories()

    def _setup_tokenizer(self) -> None:
        """Load Granite Docling tokenizer with local caching."""
        try:
            self.logger.info(
                f"Loading tokenizer: {self.config.chunking.tokenizer_model}"
            )
            cache_dir = (
                Path(self.config.docling.artifacts_path).resolve() / "tokenizers"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.chunking.tokenizer_model,
                trust_remote_code=True,
                cache_dir=str(cache_dir),
            )
            self.logger.info("Granite Docling tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            self.logger.info("Falling back to basic tokenization")
            self.tokenizer = None

    def _setup_chunker(self) -> None:
        """Initialize hierarchical document chunker."""
        try:
            self.chunker = HierarchicalChunker()
            self.logger.info("Document chunker initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize chunker: {e}")
            raise

    def _setup_vector_db(self) -> None:
        """Initialize Milvus vector database connection."""
        try:
            self.vector_db = MilvusVectorDB()
            self.logger.info("Milvus vector database initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            raise

    def _setup_image_processor(self) -> None:
        """Initialize image extraction and annotation processor."""
        try:
            self.image_processor = ImageProcessor()
            self.logger.info("Image processor initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize image processor: {e}")
            raise

    def _setup_docling(self) -> None:
        """Configure Docling converter with vision model support for Japanese documents."""
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            artifacts_path=self.config.docling.artifacts_path,
            do_ocr=self.config.docling.enable_ocr,
            do_table_structure=self.config.docling.do_table_structure,
            generate_page_images=self.config.docling.generate_page_images,
            ocr_options=EasyOcrOptions(
                use_gpu=True,
                download_enabled=True,
                model_storage_directory=".models/EasyOCR",
            ),
        )

        if self.config.docling.do_table_structure:
            pipeline_options.table_structure_options.do_cell_matching = (
                self.config.docling.do_cell_matching
            )
        if self.config.docling.enable_vision:
            pipeline_options.do_picture_description = True
            pipeline_options.images_scale = self.config.docling.images_scale
            pipeline_options.generate_picture_images = True
            from docling.datamodel.pipeline_options import PictureDescriptionVlmOptions

            pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id=self.config.docling.vision_model_repo_id,
                prompt=self.config.docling.vision_prompt,
                generation_config={"max_new_tokens": 200, "do_sample": False},
                batch_size=8,
                scale=2,
                picture_area_threshold=0.05,
            )

            self.logger.info(
                f"Vision model enabled: {self.config.docling.vision_model_repo_id}"
            )
        else:
            pipeline_options.images_scale = self.config.docling.images_scale
            pipeline_options.generate_picture_images = True
            self.logger.info("Basic image generation enabled (vision models disabled)")
        artifacts_path = Path(self.config.docling.artifacts_path).resolve()
        artifacts_path.mkdir(parents=True, exist_ok=True)
        pipeline_options.artifacts_path = artifacts_path

        self.logger.info(f"Models will be stored in: {artifacts_path}")

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.logger.info("Docling converter initialized with vision model support")

    def _ensure_directories(self) -> None:
        """Create required output directories."""
        self.config.ensure_output_dirs()
        self.logger.debug("Output directories ensured")

    def discover_files(self, directory: Path) -> list[Path]:
        """Find supported files recursively, filtering by size limits."""
        files = []

        for file_path in directory.rglob("*"):
            if file_path.is_file() and self.config.is_supported_file(file_path):
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.docling.max_file_size_mb:
                    self.logger.warning(
                        f"Skipping {file_path}: size {file_size_mb:.1f}MB exceeds limit"
                    )
                    continue

                files.append(file_path)

        self.logger.info(f"Discovered {len(files)} supported files")
        return files

    def process_files(self, files: list[Path]) -> ProcessingResults:
        """Process files in batches with progress tracking."""
        results = ProcessingResults()
        start_time = time.time()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(files))
            batch_size = self.config.processing.batch_size
            for i in range(0, len(files), batch_size):
                batch = files[i : i + batch_size]
                batch_results = self._process_batch(batch)

                results.success_count += batch_results.success_count
                results.partial_success_count += batch_results.partial_success_count
                results.failure_count += batch_results.failure_count
                results.errors.extend(batch_results.errors)
                progress.update(task, advance=len(batch))

        results.total_time = time.time() - start_time
        self.logger.info(f"Processing completed in {results.total_time:.2f}s")
        return results

    def _process_batch(self, files: list[Path]) -> ProcessingResults:
        """Convert batch of documents via Docling with error handling."""
        results = ProcessingResults()

        try:
            conv_results = self.converter.convert_all(
                files,
                raises_on_error=False,
                max_num_pages=self.config.docling.max_num_pages,
                max_file_size=self.config.docling.max_file_size_mb * 1024 * 1024,
            )
            for conv_result in conv_results:
                if conv_result.status.name == "SUCCESS":
                    results.success_count += 1
                    self._save_document(conv_result)
                elif conv_result.status.name == "PARTIAL_SUCCESS":
                    results.partial_success_count += 1
                    self._save_document(conv_result)
                    for error in conv_result.errors:
                        error_msg = f"{conv_result.input.file}: {error.error_message}"
                        results.errors.append(error_msg)
                        self.logger.warning(error_msg)
                else:
                    results.failure_count += 1
                    error_msg = f"Failed to process {conv_result.input.file}"
                    results.errors.append(error_msg)
                    self.logger.error(error_msg)

        except Exception as e:
            results.failure_count += len(files)
            error_msg = f"Batch processing failed: {e}"
            results.errors.append(error_msg)
            self.logger.error(error_msg)

        return results

    def _process_images(self, conv_result, doc_filename) -> list:
        """Extract and store document images with vision annotations."""
        extracted_images = []
        if self.config.docling.enable_vision and conv_result.document.pictures:
            extracted_images = self.image_processor.extract_and_store_images(
                conv_result.document, doc_filename
            )
            if extracted_images:
                self.image_processor.create_image_manifest(
                    doc_filename, extracted_images
                )
                self.logger.info(
                    f"Processed {len(extracted_images)} images for {doc_filename}"
                )
        return extracted_images

    def _create_image_refs_mapping(self, extracted_images) -> dict:
        """Map image self-references to metadata for chunk enhancement."""
        image_refs = {}
        if extracted_images:
            for img in extracted_images:
                if img.get("self_ref"):
                    image_refs[img["self_ref"]] = {
                        "hash": img["hash"],
                        "filename": img["filename"],
                        "caption": img.get("caption", ""),
                        "annotations": img.get("annotations", []),
                    }
        return image_refs

    def _enhance_chunk_with_images(self, chunk_text, chunk_images) -> str:
        """Append image references and vision annotations to chunk text."""
        enhanced_text = chunk_text

        for img_info in chunk_images:
            enhanced_text += f"\n[Image: {img_info['hash']}.png"
            if img_info["caption"]:
                enhanced_text += f" - {img_info['caption']}"
            enhanced_text += "]"

            if img_info["annotations"]:
                for annotation in img_info["annotations"][:2]:
                    if isinstance(annotation, dict) and "text" in annotation:
                        enhanced_text += (
                            f"\nImage description: {annotation['text'][:200]}"
                        )

        return enhanced_text

    def _create_enhanced_chunks(
        self, chunks, extracted_images
    ) -> tuple[list[str], list[dict]]:
        """Generate text chunks enriched with image information and metadata."""
        chunk_texts = []
        chunk_metadata = []

        image_refs = self._create_image_refs_mapping(extracted_images)

        for chunk in chunks:
            if not chunk.text.strip():
                continue

            chunk_images = []
            if hasattr(chunk, "refs") and chunk.refs and image_refs:
                for ref in chunk.refs:
                    if ref in image_refs:
                        chunk_images.append(image_refs[ref])
            enhanced_text = self._enhance_chunk_with_images(chunk.text, chunk_images)

            chunk_texts.append(enhanced_text)
            chunk_metadata.append(
                {
                    "original_text": chunk.text,
                    "images": chunk_images,
                    "has_images": len(chunk_images) > 0,
                }
            )

        return chunk_texts, chunk_metadata

    def _prepare_metadata(self, file_path, chunk_texts, extracted_images) -> dict:
        """Create document metadata with processing stats and Japanese content analysis."""
        metadata = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "processed_at": datetime.now().isoformat(),
            "processing_time": time.time(),
            "num_chunks": len(chunk_texts),
            "num_images": len(extracted_images),
            "has_images": len(extracted_images) > 0,
            "vision_enabled": self.config.docling.enable_vision,
        }

        if extracted_images:
            all_annotations = []
            for img in extracted_images:
                all_annotations.extend(img.get("annotations", []))
            japanese_analysis = self.image_processor.analyze_japanese_content(
                all_annotations
            )
            metadata["japanese_content_analysis"] = japanese_analysis

        return metadata

    def _save_output_formats(
        self, conv_result, doc_filename, extracted_images, chunk_texts, metadata
    ):
        """Export document to configured formats (JSON, Markdown, JSONL)."""
        raw_dir = self.config.get_output_path(self.config.output.raw_output_dir)
        processed_dir = self.config.get_output_path(
            self.config.output.processed_output_dir
        )
        chunks_dir = self.config.get_output_path(self.config.output.chunks_output_dir)

        for output_format in self.config.output.output_formats:
            if output_format == "json":
                conv_result.document.save_as_json(raw_dir / f"{doc_filename}.json")
            elif output_format == "markdown":
                self._save_enhanced_markdown(
                    conv_result.document, doc_filename, extracted_images, processed_dir
                )
            elif output_format == "jsonl":
                self._save_chunks_jsonl(
                    doc_filename, chunk_texts, metadata, extracted_images, chunks_dir
                )

    def _save_enhanced_markdown(
        self, document, doc_filename, extracted_images, processed_dir
    ):
        """Export markdown with appended image reference section."""
        markdown_content = document.export_to_markdown()
        if extracted_images:
            image_refs = self.image_processor.get_image_references_for_text(
                extracted_images
            )
            markdown_content += f"\n\n## Extracted Images\n\n{image_refs}"

        markdown_path = processed_dir / f"{doc_filename}.md"
        with markdown_path.open("w", encoding="utf-8") as f:
            f.write(markdown_content)

    def _save_chunks_jsonl(
        self, doc_filename, chunk_texts, metadata, extracted_images, chunks_dir
    ):
        """Export chunks to JSONL format with metadata and image associations."""
        jsonl_path = chunks_dir / f"{doc_filename}_chunks.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for i, chunk_text in enumerate(chunk_texts):
                chunk_data = {
                    "document_id": doc_filename,
                    "chunk_index": i,
                    "text": chunk_text,
                    "metadata": metadata,
                    "images": extracted_images,
                }
                f.write(str(chunk_data) + "\n")

    def _save_document(self, conv_result) -> None:
        """Process document with Late Chunking, save outputs, and store in vector database."""
        doc_filename = conv_result.input.file.stem
        file_path = conv_result.input.file

        try:
            extracted_images = self._process_images(conv_result, doc_filename)
            use_late_chunking = (
                hasattr(self.config.chunking, "use_late_chunking")
                and self.config.chunking.use_late_chunking
            )

            if use_late_chunking:
                full_text = conv_result.document.export_to_markdown()
                metadata = self._prepare_metadata(file_path, [], extracted_images)
                success = self.vector_db.insert_document_with_late_chunking(
                    doc_id=doc_filename,
                    full_document=full_text,
                    metadata=metadata,
                    max_chunk_length=800,
                )

                if success:
                    self.logger.info(
                        f"Stored document using Late Chunking for {doc_filename}"
                    )
                else:
                    self.logger.warning(
                        f"Failed to store document with Late Chunking for {doc_filename}"
                    )

                chunks = list(self.chunker.chunk(conv_result.document))
                chunk_texts, chunk_metadata = self._create_enhanced_chunks(
                    chunks, extracted_images
                )
            else:
                chunks = list(self.chunker.chunk(conv_result.document))
                chunk_texts, chunk_metadata = self._create_enhanced_chunks(
                    chunks, extracted_images
                )
                metadata = self._prepare_metadata(
                    file_path, chunk_texts, extracted_images
                )
                if chunk_texts:
                    success = self.vector_db.insert_document(
                        doc_id=doc_filename,
                        text_chunks=chunk_texts,
                        metadata=metadata,
                        chunk_metadata=chunk_metadata,
                    )
                    if success:
                        self.logger.info(
                            f"Stored {len(chunk_texts)} traditional chunks in vector DB for {doc_filename}"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to store chunks in vector DB for {doc_filename}"
                        )

            self._save_output_formats(
                conv_result, doc_filename, extracted_images, chunk_texts, metadata
            )

            self.logger.debug(
                f"Saved document with {len(extracted_images)} images: {doc_filename}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save document {doc_filename}: {e}")
            raise
