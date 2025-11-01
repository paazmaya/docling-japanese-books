"""Image processing and storage utilities for Japanese books."""

import base64
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from docling_core.types.doc.document import (
    DoclingDocument,
    PictureDescriptionData,
    PictureItem,
)
from PIL import Image

from .config import config


class ImageProcessor:
    """Handles image extraction, storage, and annotation processing."""

    def __init__(self) -> None:
        """Initialize the image processor."""
        self.logger = logging.getLogger(__name__)
        self.config = config

    def extract_and_store_images(
        self, document: DoclingDocument, doc_id: str
    ) -> list[dict[str, str]]:
        """Extract images from document and store them separately.

        This method processes all images found in a DoclingDocument, extracts them,
        saves them to the filesystem with SHA-256 hashed filenames, and collects
        associated metadata including captions and vision model annotations.

        Args:
            document: The DoclingDocument containing images to extract.
            doc_id: Unique identifier for the document, used for directory naming.

        Returns:
            List of dictionaries containing image metadata:
                - hash: SHA-256 hash of the image content
                - filename: Generated filename (hash + .png extension)
                - path: Full path to the saved image file
                - width: Image width in pixels
                - height: Image height in pixels
                - caption: Caption text from the document
                - annotations: List of vision model descriptions/annotations

        Raises:
            OSError: If there are issues creating directories or saving files.
        """
        images_dir = self.config.get_output_path(self.config.output.images_output_dir)
        doc_images_dir = images_dir / doc_id
        doc_images_dir.mkdir(parents=True, exist_ok=True)

        extracted_images = []

        for i, picture in enumerate(document.pictures):
            try:
                image_info = self._process_picture(
                    picture, doc_id, i, doc_images_dir, document
                )
                if image_info:
                    extracted_images.append(image_info)
            except Exception as e:
                self.logger.error(
                    f"Failed to process picture {i} in document {doc_id}: {e}"
                )

        self.logger.info(
            f"Extracted {len(extracted_images)} images from document {doc_id}"
        )
        return extracted_images

    def _process_picture(
        self,
        picture: PictureItem,
        doc_id: str,
        index: int,
        output_dir: Path,
        document: DoclingDocument,
    ) -> Optional[dict[str, str]]:
        """Process a single picture item and save it to the filesystem.

        This method handles the extraction, hashing, and storage of a single image
        from a document. It generates a SHA-256 hash for consistent naming,
        extracts metadata, and collects any available annotations.

        Args:
            picture: The PictureItem from the document to process.
            doc_id: Document identifier for reference tracking.
            index: Sequential index of this picture within the document.
            output_dir: Directory where the image file will be saved.
            document: Parent document for caption text extraction.

        Returns:
            Dictionary containing image metadata if successful, None if processing fails.
            Keys include: document_id, image_index, filename, file_path, hash,
            width, height, caption, and annotations.

        Raises:
            Exception: Re-raised after logging if image processing fails.
        """
        try:
            # Extract image data
            image_data = self._extract_image_data(picture)
            if not image_data:
                return None

            # Generate SHA-256 hash of image content for consistent filename
            image_hash = hashlib.sha256(image_data).hexdigest()
            filename = f"{image_hash}.png"
            image_path = output_dir / filename

            # Save image
            with image_path.open("wb") as f:
                f.write(image_data)

            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                self.logger.warning(f"Could not get image dimensions: {e}")
                width, height = 0, 0

            # Extract annotations (vision model descriptions)
            annotations = self._extract_annotations(picture)

            # Get caption text
            caption = (
                picture.caption_text(doc=document)
                if hasattr(picture, "caption_text")
                else ""
            )

            return {
                "document_id": doc_id,
                "image_index": index,
                "filename": filename,
                "file_path": str(image_path),
                "relative_path": f"{doc_id}/{filename}",
                "width": width,
                "height": height,
                "file_size": len(image_data),
                "hash": image_hash,
                "caption": caption,
                "annotations": annotations,
                "self_ref": picture.self_ref,
            }

        except Exception as e:
            self.logger.error(f"Failed to process picture {index}: {e}")
            return None

    def _extract_image_data(self, picture: PictureItem) -> Optional[bytes]:
        """Extract binary image data from picture item."""
        try:
            if hasattr(picture, "image") and hasattr(picture.image, "uri"):
                uri = picture.image.uri

                # Handle data URI (base64 encoded images)
                if uri.startswith("data:image/"):
                    # Extract base64 data after the comma
                    if "," in uri:
                        base64_data = uri.split(",", 1)[1]
                        return base64.b64decode(base64_data)

                # Handle file URI
                elif uri.startswith("file://"):
                    file_path = Path(urlparse(uri).path)
                    if file_path.exists():
                        return file_path.read_bytes()

                self.logger.warning(f"Unsupported image URI format: {uri[:100]}...")
                return None

        except Exception as e:
            self.logger.error(f"Failed to extract image data: {e}")
            return None

    def _extract_annotations(self, picture: PictureItem) -> list[dict[str, str]]:
        """Extract vision model annotations from picture."""
        annotations = []

        try:
            for annotation in picture.annotations:
                if isinstance(annotation, PictureDescriptionData):
                    annotations.append(
                        {
                            "model": annotation.provenance,
                            "text": annotation.text,
                            "confidence": getattr(annotation, "confidence", None),
                        }
                    )
        except Exception as e:
            self.logger.warning(f"Failed to extract annotations: {e}")

        return annotations

    def create_image_manifest(self, doc_id: str, images: list[dict[str, str]]) -> None:
        """Create a JSON manifest file for all images in a document."""
        images_dir = self.config.get_output_path(self.config.output.images_output_dir)
        doc_images_dir = images_dir / doc_id

        manifest = {
            "document_id": doc_id,
            "total_images": len(images),
            "images": images,
            "created_at": self._get_timestamp(),
        }

        manifest_path = doc_images_dir / f"{doc_id}_manifest.json"

        try:
            import json

            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Created image manifest: {manifest_path}")

        except Exception as e:
            self.logger.error(f"Failed to create image manifest: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_legacy_image_references_for_text(self, images: list[dict[str, str]]) -> str:
        """Generate text references for images that can be included in markdown."""
        if not images:
            return ""

        references = []
        for img in images:
            # Create markdown-style image reference
            caption = img.get("caption") or f"Image {img['image_index']}"
            ref = f"![{caption}]({img['relative_path']})"

            # Add annotations as comments
            if img.get("annotations"):
                for annotation in img["annotations"]:
                    ref += f"\n<!-- Vision Model ({annotation['model']}): {annotation['text']} -->"

            references.append(ref)

        return "\n\n".join(references)

    def analyze_japanese_content(self, annotations: list[str]) -> dict:
        """Analyze annotations for Japanese content patterns."""
        if not annotations:
            return {
                "has_japanese": False,
                "confidence": 0.0,
                "japanese_indicators": [],
                "content_summary": "No annotations available",
            }

        all_text = " ".join(annotations)

        # Japanese script detection patterns
        hiragana_count = len(re.findall(r"[\u3040-\u309F]", all_text))
        katakana_count = len(re.findall(r"[\u30A0-\u30FF]", all_text))
        kanji_count = len(re.findall(r"[\u4E00-\u9FAF]", all_text))

        # Japanese cultural/contextual indicators
        indicators = []
        japanese_patterns = [
            (r"(?:san|chan|kun|sama)", "Japanese honorifics"),
            (r"(?:sushi|ramen|tempura|yakitori)", "Japanese food terms"),
            (r"(?:kimono|yukata|obi)", "Japanese clothing"),
            (r"(?:temple|shrine|torii|pagoda)", "Japanese architecture"),
            (r"(?:sakura|cherry blossom)", "Japanese nature"),
            (r"(?:manga|anime)", "Japanese media"),
        ]

        for pattern, description in japanese_patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                indicators.append(description)

        # Calculate confidence
        total_chars = len(all_text)
        japanese_chars = hiragana_count + katakana_count + kanji_count
        script_confidence = japanese_chars / total_chars if total_chars > 0 else 0
        indicator_confidence = min(len(indicators) * 0.2, 1.0)
        overall_confidence = (script_confidence * 0.7) + (indicator_confidence * 0.3)

        return {
            "has_japanese": overall_confidence > 0.3,
            "confidence": round(overall_confidence, 3),
            "script_analysis": {
                "hiragana_count": hiragana_count,
                "katakana_count": katakana_count,
                "kanji_count": kanji_count,
                "total_japanese_chars": japanese_chars,
            },
            "japanese_indicators": indicators,
            "content_summary": f"Analyzed {len(annotations)} image annotations",
        }

    def get_image_references_for_text(self, extracted_images: list[dict]) -> str:
        """Generate markdown-formatted image references for text output."""
        if not extracted_images:
            return "No images found in document."

        references = []
        for img in extracted_images:
            image_path = img["path"]
            annotations = img.get("annotations", [])

            # Create a clean reference
            ref_lines = [f"### Image: {image_path.name}"]
            ref_lines.append(f"**Path:** `{image_path}`")

            if annotations:
                ref_lines.append("**Descriptions:**")
                for i, annotation in enumerate(annotations, 1):
                    # Truncate long annotations
                    display_annotation = (
                        annotation[:200] + "..."
                        if len(annotation) > 200
                        else annotation
                    )
                    ref_lines.append(f"{i}. {display_annotation}")
            else:
                ref_lines.append("*No annotations generated*")

            references.append("\n".join(ref_lines))

        return "\n\n".join(references)

    def _check_japanese_indicators(self, text: str, analysis: dict[str, any]) -> None:
        """Check text for Japanese writing system indicators."""
        japanese_indicators = [
            "japanese",
            "hiragana",
            "katakana",
            "kanji",
            "漢字",
            "ひらがな",
            "カタカナ",
            "文字",
            "テキスト",
            "日本語",
            "和文",
        ]
        writing_systems = ["hiragana", "katakana", "kanji"]

        for indicator in japanese_indicators:
            if indicator in text:
                analysis["contains_japanese_text"] = True
                if indicator in writing_systems:
                    analysis["writing_system_detected"].append(indicator)

    def _check_layout_orientation(self, text: str, analysis: dict[str, any]) -> None:
        """Check text for layout and orientation indicators."""
        layout_indicators = [
            "vertical",
            "horizontal",
            "縦書き",
            "横書き",
            "right-to-left",
        ]
        if any(indicator in text.lower() for indicator in layout_indicators):
            analysis["layout_cues"] = True

    def _check_cultural_elements(self, text: str, analysis: dict[str, any]) -> None:
        """Check text for Japanese cultural elements and context."""
