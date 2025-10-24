"""Model downloader for HuggingFace models with progress tracking and local caching."""

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
    """Model download operation result with path and error information."""

    success: bool
    model_path: Optional[Path] = None
    error: Optional[str] = None


class ModelDownloader:
    """HuggingFace model downloader with progress tracking and verification."""

    def __init__(self) -> None:
        """Initialize downloader with model cache directory and HF API client."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.console = Console()

        self.models_dir = Path(self.config.docling.artifacts_path).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Models directory: {self.models_dir}")

        self.downloaded_count = 0
        self.total_models = 3
        self.hf_api = HfApi()

    def _verify_model_exists(self, repo_id: str) -> bool:
        """Check if model repository exists on HuggingFace Hub."""
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
        """Download model from HuggingFace Hub with verification and progress updates."""
        progress.update(
            task_id, description=f"{emoji} Verifying {model_type}: {repo_id}"
        )

        if not self._verify_model_exists(repo_id):
            error_msg = f"Model {repo_id} not found on HuggingFace Hub"
            progress.update(task_id, description=f"❌ {model_type.title()} not found")
            return DownloadResult(success=False, error=error_msg)

        progress.update(
            task_id, description=f"{emoji} Downloading {model_type}: {repo_id}"
        )
        try:
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
        """Download Granite Docling tokenizer model."""
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
        """Download BGE-M3 multilingual embedding model."""
        return self._download_model(
            repo_id=self.config.chunking.embedding_model,
            model_type="embedding",
            cache_subdir="embeddings",
            emoji="🧮",
            progress=progress,
            task_id=task_id,
        )

    def download_vision_model(self, progress: Progress, task_id: int) -> DownloadResult:
        """Download Granite Vision model for image descriptions."""
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
        """Download all configured models with progress display and error handling."""
        self.logger.info("🚀 Starting model downloads...")

        results = {}
        self.downloaded_count = 0

        if not self.config.docling.enable_vision:
            self.total_models = 2

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task(
                "📦 Preparing downloads...", total=self.total_models
            )
            progress.update(task_id, advance=0)
            results["tokenizer"] = self.download_tokenizer(progress, task_id)
            if results["tokenizer"].success:
                progress.update(task_id, advance=1)

            results["embedding"] = self.download_embedding_model(progress, task_id)
            if results["embedding"].success:
                progress.update(task_id, advance=1)

            results["vision"] = self.download_vision_model(progress, task_id)
            if results["vision"].success or not self.config.docling.enable_vision:
                progress.update(task_id, advance=1)
            progress.update(
                task_id,
                description=f"🎯 Downloads completed ({self.downloaded_count}/{self.total_models})",
            )

        successful = [name for name, result in results.items() if result.success]
        failed = [name for name, result in results.items() if not result.success]

        self.logger.info(f"✅ Successfully downloaded: {', '.join(successful)}")
        if failed:
            self.logger.error(f"❌ Failed to download: {', '.join(failed)}")
        else:
            self.logger.info("🎉 All models downloaded successfully!")

        return results

    def check_models_exist(self) -> dict[str, bool]:
        """Verify which models exist in local cache."""
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
        """Return configured model repository IDs."""
        return {
            "tokenizer": self.config.chunking.tokenizer_model,
            "embedding": self.config.chunking.embedding_model,
            "vision": self.config.docling.vision_model_repo_id
            if self.config.docling.enable_vision
            else "disabled",
        }
