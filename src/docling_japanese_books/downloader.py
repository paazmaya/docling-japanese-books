"""
Model downloader for Docling Japanese Books.

This module handles downloading and caching models locally.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .config import config


@dataclass
class DownloadResult:
    """Result of a model download operation."""

    success: bool
    model_path: Optional[Path] = None
    error: Optional[str] = None


class ModelDownloader:
    """Downloads and manages models locally."""

    def __init__(self) -> None:
        """Initialize the model downloader."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.console = Console()

        # Ensure models directory exists
        self.models_dir = Path(self.config.docling.artifacts_path).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Models directory: {self.models_dir}")

        # Track download progress
        self.downloaded_count = 0
        self.total_models = 3  # tokenizer, embedding, vision

    def download_tokenizer(self, progress: Progress, task_id: int) -> DownloadResult:
        """Download the Granite Docling tokenizer."""
        model_name = self.config.chunking.tokenizer_model
        progress.update(task_id, description=f"📝 Downloading tokenizer: {model_name}")

        try:
            # Download to local cache first
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=self.models_dir / "tokenizers",
            )

            model_path = self.models_dir / "tokenizers" / model_name.replace("/", "--")
            self.downloaded_count += 1

            progress.update(
                task_id,
                description=f"✅ Tokenizer downloaded ({self.downloaded_count}/{self.total_models})",
            )
            self.logger.info(f"✅ Tokenizer downloaded successfully to {model_path}")
            return DownloadResult(success=True, model_path=model_path)

        except Exception as e:
            error_msg = f"Failed to download tokenizer {model_name}: {e}"
            progress.update(task_id, description="❌ Tokenizer failed")
            self.logger.error(error_msg)
            return DownloadResult(success=False, error=error_msg)

    def download_embedding_model(
        self, progress: Progress, task_id: int
    ) -> DownloadResult:
        """Download the sentence transformer embedding model."""
        model_name = self.config.chunking.embedding_model
        progress.update(task_id, description=f"🧮 Downloading embedding: {model_name}")

        try:
            # Download to local cache
            SentenceTransformer(
                model_name, cache_folder=str(self.models_dir / "embeddings")
            )

            model_path = self.models_dir / "embeddings" / model_name.replace("/", "--")
            self.downloaded_count += 1

            progress.update(
                task_id,
                description=f"✅ Embedding downloaded ({self.downloaded_count}/{self.total_models})",
            )
            self.logger.info(
                f"✅ Embedding model downloaded successfully to {model_path}"
            )
            return DownloadResult(success=True, model_path=model_path)

        except Exception as e:
            error_msg = f"Failed to download embedding model {model_name}: {e}"
            progress.update(task_id, description="❌ Embedding failed")
            self.logger.error(error_msg)
            return DownloadResult(success=False, error=error_msg)

    def download_vision_model(self, progress: Progress, task_id: int) -> DownloadResult:
        """Download the vision model for image description."""
        if not self.config.docling.enable_vision:
            progress.update(task_id, description="⏭️  Vision model disabled, skipping")
            self.logger.info("Vision models disabled, skipping download")
            return DownloadResult(success=True)

        model_name = self.config.docling.vision_model_repo_id
        progress.update(task_id, description=f"👁️  Downloading vision: {model_name}")

        try:
            # Use huggingface_hub to download the entire model
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.models_dir / "vision",
                local_files_only=False,
            )

            self.downloaded_count += 1
            progress.update(
                task_id,
                description=f"✅ Vision downloaded ({self.downloaded_count}/{self.total_models})",
            )
            self.logger.info(f"✅ Vision model downloaded successfully to {model_path}")
            return DownloadResult(success=True, model_path=Path(model_path))

        except Exception as e:
            error_msg = f"Failed to download vision model {model_name}: {e}"
            progress.update(task_id, description="❌ Vision failed")
            self.logger.error(error_msg)
            return DownloadResult(success=False, error=error_msg)

    def download_all_models(self) -> dict[str, DownloadResult]:
        """Download all required models with progress tracking."""
        self.logger.info("🚀 Starting model downloads...")

        results = {}
        self.downloaded_count = 0  # Reset counter

        # Adjust total count based on vision model setting
        if not self.config.docling.enable_vision:
            self.total_models = 2

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            # Create progress task
            task_id = progress.add_task(
                "� Preparing downloads...", total=self.total_models
            )

            # Download tokenizer
            progress.update(task_id, advance=0)
            results["tokenizer"] = self.download_tokenizer(progress, task_id)
            if results["tokenizer"].success:
                progress.update(task_id, advance=1)

            # Download embedding model
            results["embedding"] = self.download_embedding_model(progress, task_id)
            if results["embedding"].success:
                progress.update(task_id, advance=1)

            # Download vision model
            results["vision"] = self.download_vision_model(progress, task_id)
            if results["vision"].success or not self.config.docling.enable_vision:
                progress.update(task_id, advance=1)

            # Final update
            progress.update(
                task_id,
                description=f"🎯 Downloads completed ({self.downloaded_count}/{self.total_models})",
            )

        # Summary
        successful = [name for name, result in results.items() if result.success]
        failed = [name for name, result in results.items() if not result.success]

        self.logger.info(f"✅ Successfully downloaded: {', '.join(successful)}")
        if failed:
            self.logger.error(f"❌ Failed to download: {', '.join(failed)}")
        else:
            self.logger.info("🎉 All models downloaded successfully!")

        return results

    def check_models_exist(self) -> dict[str, bool]:
        """Check which models are already downloaded."""
        status = {}

        # Check tokenizer
        tokenizer_path = self.models_dir / "tokenizers"
        status["tokenizer"] = tokenizer_path.exists() and any(tokenizer_path.iterdir())

        # Check embedding model
        embedding_path = self.models_dir / "embeddings"
        status["embedding"] = embedding_path.exists() and any(embedding_path.iterdir())

        # Check vision model
        vision_path = self.models_dir / "vision"
        status["vision"] = vision_path.exists() and any(vision_path.iterdir())

        return status

    def get_model_info(self) -> dict[str, str]:
        """Get information about configured models."""
        return {
            "tokenizer": self.config.chunking.tokenizer_model,
            "embedding": self.config.chunking.embedding_model,
            "vision": self.config.docling.vision_model_repo_id
            if self.config.docling.enable_vision
            else "disabled",
        }
