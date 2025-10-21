"""
Model downloader for Docling Japanese Books.

This module handles downloading and caching models locally using HuggingFace Hub.
All models are downloaded via snapshot_download() for consistency and reliability.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

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

        # Initialize HuggingFace API client
        self.hf_api = HfApi()

    def _verify_model_exists(self, repo_id: str) -> bool:
        """Verify that a model exists on HuggingFace Hub."""
        try:
            self.hf_api.repo_info(repo_id=repo_id)
            return True
        except Exception as e:
            self.logger.warning(f"Could not verify model {repo_id}: {e}")
            return False

    def _download_model(
        self,
        repo_id: str,
        model_type: str,
        cache_subdir: str,
        emoji: str,
        progress: Progress,
        task_id: int,
    ) -> DownloadResult:
        """Generic model download function using HuggingFace Hub."""
        progress.update(
            task_id, description=f"{emoji} Verifying {model_type}: {repo_id}"
        )

        # Verify model exists on HuggingFace Hub
        if not self._verify_model_exists(repo_id):
            error_msg = f"Model {repo_id} not found on HuggingFace Hub"
            progress.update(task_id, description=f"❌ {model_type.title()} not found")
            return DownloadResult(success=False, error=error_msg)

        progress.update(
            task_id, description=f"{emoji} Downloading {model_type}: {repo_id}"
        )
        try:
            # Use HuggingFace Hub to download the entire model
            cache_dir = self.models_dir / cache_subdir
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                local_files_only=False,
            )

            self.downloaded_count += 1
            progress.update(
                task_id,
                description=f"✅ {model_type.title()} downloaded ({self.downloaded_count}/{self.total_models})",
            )
            self.logger.info(
                f"✅ {model_type.title()} downloaded successfully to {model_path}"
            )
            return DownloadResult(success=True, model_path=Path(model_path))

        except Exception as e:
            error_msg = f"Failed to download {model_type} {repo_id}: {e}"
            progress.update(task_id, description=f"❌ {model_type.title()} failed")
            self.logger.error(error_msg)
            return DownloadResult(success=False, error=error_msg)

    def download_tokenizer(self, progress: Progress, task_id: int) -> DownloadResult:
        """Download the Granite Docling tokenizer using HuggingFace Hub."""
        return self._download_model(
            repo_id=self.config.chunking.tokenizer_model,
            model_type="tokenizer",
            cache_subdir="tokenizers",
            emoji="📝",
            progress=progress,
            task_id=task_id,
        )

    def download_embedding_model(
        self, progress: Progress, task_id: int
    ) -> DownloadResult:
        """Download the sentence transformer embedding model using HuggingFace Hub."""
        return self._download_model(
            repo_id=self.config.chunking.embedding_model,
            model_type="embedding",
            cache_subdir="embeddings",
            emoji="🧮",
            progress=progress,
            task_id=task_id,
        )

    def download_vision_model(self, progress: Progress, task_id: int) -> DownloadResult:
        """Download the vision model for image description."""
        if not self.config.docling.enable_vision:
            progress.update(task_id, description="⏭️  Vision model disabled, skipping")
            self.logger.info("Vision models disabled, skipping download")
            return DownloadResult(success=True)

        return self._download_model(
            repo_id=self.config.docling.vision_model_repo_id,
            model_type="vision",
            cache_subdir="vision",
            emoji="👁️",
            progress=progress,
            task_id=task_id,
        )

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
        models_to_check = [
            ("tokenizer", "tokenizers"),
            ("embedding", "embeddings"),
            ("vision", "vision"),
        ]

        status = {}
        for model_name, cache_subdir in models_to_check:
            model_path = self.models_dir / cache_subdir
            status[model_name] = model_path.exists() and any(model_path.iterdir())

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
