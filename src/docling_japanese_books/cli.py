"""Command-line interface for the document processing pipeline."""

import logging
import sys
from pathlib import Path

# Built-in typing support for Python 3.9+
import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .config import config
from .downloader import ModelDownloader
from .processor import DocumentProcessor


def setup_logging(level: str = "INFO") -> None:
    """Set up logging with rich formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def display_config_panel(console: Console, directory: Path) -> None:
    """Display configuration panel."""
    console.print()
    # Determine vision status and model
    vision_status = "Enabled" if config.docling.enable_vision else "Disabled"
    vision_display = (
        f"{vision_status} ({config.docling.vision_model})"
        if config.docling.enable_vision
        else "Disabled"
    )

    # Database display based on deployment mode
    db_mode = config.database.deployment_mode
    if db_mode == "cloud":
        db_display = f"Zilliz Cloud ({config.database.zilliz_cluster_id or 'cluster'})"
        db_path_display = f"Cloud: [bright_cyan]{config.database.zilliz_cloud_uri or 'Not configured'}[/bright_cyan]"
    else:
        db_display = "Milvus Lite (Local)"
        db_path_display = (
            f"Local: [bright_black]{config.database.milvus_uri}[/bright_black]"
        )

    console.print(
        Panel.fit(
            f"🚀 [bold blue]Docling Japanese Books Processor[/bold blue]\n\n"
            f"📁 Input Directory: [cyan]{directory.absolute()}[/cyan]\n"
            f"🎯 Output Directory: [cyan]{config.output.output_base_dir}[/cyan]\n"
            f"🔧 Granite Model: [yellow]{config.chunking.tokenizer_model}[/yellow]\n"
            f"🧮 Embedding Model: [yellow]{config.chunking.embedding_model}[/yellow]\n"
            f"👁️  Vision Model: [bright_magenta]{vision_display}[/bright_magenta]\n"
            f"🖼️  Image Storage: [bright_cyan]{config.output.images_output_dir}[/bright_cyan]\n"
            f"📊 Vector Database: [green]{db_display}[/green]\n"
            f"💾 Database: {db_path_display}\n"
            f"📝 Output Formats: [magenta]{', '.join(config.output.output_formats)}[/magenta]\n"
            f"🔄 Batch Size: [bright_black]{config.processing.batch_size}[/bright_black]\n"
            f"🧵 Max Workers: [bright_black]{config.processing.max_workers}[/bright_black]",
            title="Configuration",
            border_style="blue",
        )
    )


def discover_and_display_files(
    processor: DocumentProcessor, directory: Path, console: Console
) -> list[Path]:
    """Discover files and display summary."""
    console.print()
    console.print("🔍 [yellow]Discovering files...[/yellow]")

    files = processor.discover_files(directory)
    if not files:
        console.print(f"⚠️ [yellow]No supported files found in {directory}[/yellow]")
        return files

    console.print(f"📄 Found {len(files)} supported files")

    # Show file types summary
    file_types = {}
    for file_path in files:
        ext = file_path.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    type_summary = ", ".join(
        [f"{ext}: {count}" for ext, count in sorted(file_types.items())]
    )
    console.print(f"📋 File types: {type_summary}")

    return files


def handle_dry_run(files: list[Path], console: Console) -> None:
    """Handle dry run display."""
    console.print()
    console.print(
        "🔍 [yellow]Dry run - showing files that would be processed:[/yellow]"
    )
    for i, file_path in enumerate(files[:10], 1):  # Show first 10 files
        console.print(f"  {i:3d}. {file_path.name}")
    if len(files) > 10:
        console.print(f"  ... and {len(files) - 10} more files")
    console.print()
    console.print("💡 Run without --dry-run to actually process the files")


def display_results(results, files: list[Path], console: Console) -> None:
    """Display processing results."""
    console.print()
    console.print(
        Panel.fit(
            f"✅ [bold green]Processing Complete![/bold green]\n\n"
            f"📄 Total Files: [cyan]{len(files)}[/cyan]\n"
            f"✅ Successfully Processed: [green]{results.success_count}[/green]\n"
            f"⚠️ Partial Success: [yellow]{results.partial_success_count}[/yellow]\n"
            f"❌ Failed: [red]{results.failure_count}[/red]\n"
            f"⏱️ Processing Time: [blue]{results.total_time:.2f}s[/blue]\n"
            f"📊 Average per File: [bright_black]{results.total_time / len(files):.2f}s[/bright_black]",
            title="Results Summary",
            border_style="green",
        )
    )

    if results.failure_count > 0:
        console.print("\n❌ [red]Failed files:[/red]")
        for error in results.errors[:5]:  # Show first 5 errors
            console.print(f"  • {error}")
        if len(results.errors) > 5:
            console.print(f"  ... and {len(results.errors) - 5} more errors")


@click.group()
def cli() -> None:
    """Docling Japanese Books - Document processing for Japanese books and LLM training."""
    pass


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without actually processing",
)
def process(directory: Path, verbose: bool, dry_run: bool) -> None:
    """Process documents in the specified DIRECTORY using hardcoded Docling settings.

    This tool processes all supported document files in the given directory
    and its subdirectories using pre-configured Docling settings optimized
    for Japanese documents and LLM training data preparation.

    Supported formats: PDF, DOCX, PPTX, HTML, Markdown, TXT, and images.
    """
    console = Console()

    # Set up logging
    log_level = "DEBUG" if verbose else config.processing.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Validate directory
    if not directory.is_dir():
        console.print(f"❌ [red]Error:[/red] {directory} is not a directory")
        sys.exit(1)

    # Display configuration
    display_config_panel(console, directory)

    # Initialize processor
    try:
        processor = DocumentProcessor()
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        console.print(f"❌ [red]Error:[/red] Failed to initialize processor: {e}")
        sys.exit(1)

    # Discover files
    try:
        files = discover_and_display_files(processor, directory, console)
        if not files:
            return
    except Exception as e:
        logger.error(f"Failed to discover files: {e}")
        console.print(f"❌ [red]Error:[/red] Failed to discover files: {e}")
        sys.exit(1)

    # Handle dry run
    if dry_run:
        handle_dry_run(files, console)
        return

    # Process files
    console.print()
    console.print("⚙️ [green]Starting document processing...[/green]")

    try:
        results = processor.process_files(files)
        display_results(results, files, console)
    except KeyboardInterrupt:
        console.print("\n⏹️ [yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        console.print(f"\n❌ [red]Error:[/red] Processing failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-download even if models exist",
)
def download(verbose: bool, force: bool) -> None:
    """Download all required models to local .models directory."""
    console = Console()
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    console.print(
        "\n🚀 [bold blue]Docling Japanese Books - Model Downloader[/bold blue]"
    )
    console.print(
        Panel(
            _get_download_config_panel(),
            title="📋 Download Configuration",
            border_style="blue",
        )
    )

    try:
        downloader = ModelDownloader()

        if not force and _check_existing_models(console, downloader):
            return

        results = downloader.download_all_models()
        _display_download_results(console, results)

    except KeyboardInterrupt:
        console.print("\n⏹️ [yellow]Download interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        console.print(f"\n❌ [red]Error:[/red] Download failed: {e}")
        sys.exit(1)


def _check_existing_models(console: Console, downloader: ModelDownloader) -> bool:
    """Check if models exist and return True if all exist (skip download)."""
    existing_models = downloader.check_models_exist()
    if any(existing_models.values()):
        console.print("\n📦 [yellow]Existing models found:[/yellow]")
        for model_name, exists in existing_models.items():
            status = "✅ Downloaded" if exists else "❌ Missing"
            console.print(f"  • {model_name}: {status}")

        if all(existing_models.values()):
            console.print("\n✅ [green]All models are already downloaded![/green]")
            console.print("Use --force to re-download.")
            return True
    return False


def _display_download_results(console: Console, results: dict) -> None:
    """Display download results and summary."""
    console.print("\n📊 [bold]Download Results:[/bold]")
    for model_name, result in results.items():
        if result.success:
            console.print(f"  ✅ {model_name}: Downloaded successfully")
            if result.model_path:
                console.print(f"     📁 Path: {result.model_path}")
        else:
            console.print(f"  ❌ {model_name}: Failed")
            if result.error:
                console.print(f"     💥 Error: {result.error}")

    successful = sum(1 for result in results.values() if result.success)
    total = len(results)

    if successful == total:
        console.print(
            f"\n🎉 [bold green]All {total} models downloaded successfully![/bold green]"
        )
    else:
        console.print(
            f"\n⚠️ [bold yellow]{successful}/{total} models downloaded successfully[/bold yellow]"
        )


def _get_download_config_panel() -> str:
    """Generate configuration panel content for downloads."""
    downloader = ModelDownloader()
    model_info = downloader.get_model_info()

    content = []
    content.append(
        f"📁 Models Directory: {Path(config.docling.artifacts_path).resolve()}"
    )
    content.append("")
    content.append("🤖 Models to Download:")
    content.append(f"  • Tokenizer: {model_info['tokenizer']}")
    content.append(f"  • Embedding: {model_info['embedding']}")
    content.append(f"  • Vision: {model_info['vision']}")

    return "\n".join(content)


@cli.command()
@click.argument("query_text")
@click.option(
    "--limit",
    "-l",
    default=5,
    help="Maximum number of results to return",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def search(query_text: str, limit: int, verbose: bool) -> None:
    """Search the vector database for similar content."""
    from .query import QueryInterface

    console = Console()
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    try:
        query_interface = QueryInterface()
        results = query_interface.search(query_text, limit=limit)

        if not results:
            console.print("No results found.")
            return

        console.print(f"\n🔍 Found {len(results)} results for: '{query_text}'\n")

        for i, result in enumerate(results, 1):
            console.print(
                f"[bold cyan]{i}.[/bold cyan] [bold]{result['document_name']}[/bold]"
            )
            console.print(f"   Score: {result['score']:.4f}")
            console.print(f"   {result['content'][:200]}...")
            console.print()

    except Exception as e:
        console.print(f"❌ Search failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="embedding_evaluation_results.json",
    help="Output file for evaluation results",
)
@click.option(
    "--documents",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    help="JSON file with documents to evaluate (uses sample docs if not provided)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def evaluate(output: Path, documents: Path, verbose: bool) -> None:
    """Evaluate and compare embedding approaches for Japanese documents."""
    import json

    from .embedding_evaluation import (
        EmbeddingEvaluator,
        create_sample_japanese_documents,
    )

    console = Console()
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    try:
        evaluator = EmbeddingEvaluator()

        # Load documents
        if documents:
            console.print(f"📖 Loading documents from: {documents}")
            with documents.open("r", encoding="utf-8") as f:
                docs = json.load(f)
        else:
            console.print("📖 Using sample Japanese documents for evaluation...")
            docs = create_sample_japanese_documents()

        console.print(f"🔬 Evaluating {len(docs)} documents...")
        console.print("📊 Comparing Traditional vs Late Chunking approaches")
        console.print("🧠 Testing BGE-M3 multilingual model vs sentence-transformers")
        console.print()

        # Run evaluation
        evaluator.run_comparison_study(docs, output_path=output)

        console.print(f"✅ Evaluation complete! Results saved to: {output}")

    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


def _display_current_config(console: Console) -> None:
    """Display current database configuration."""
    current_mode = config.database.deployment_mode
    console.print(f"📊 Current mode: [yellow]{current_mode}[/yellow]")

    if current_mode == "cloud":
        console.print(
            f"🌐 Current cloud URI: [cyan]{config.database.zilliz_cloud_uri or 'Not set'}[/cyan]"
        )
        console.print(
            f"🔑 API key configured: [green]{'Yes' if config.database.zilliz_api_key else 'No'}[/green]"
        )
        console.print(
            f"🏷️  Cluster ID: [bright_black]{config.database.zilliz_cluster_id or 'Not set'}[/bright_black]"
        )
    else:
        console.print(f"💾 Local database: [cyan]{config.database.milvus_uri}[/cyan]")

    console.print()


def _show_cloud_instructions(console: Console) -> None:
    """Show Zilliz Cloud configuration instructions."""
    console.print("💡 [bold]Zilliz Cloud Configuration:[/bold]")
    console.print("Set environment variables or use command options:")
    console.print("  [cyan]export MILVUS_DEPLOYMENT_MODE=cloud[/cyan]")
    console.print(
        "  [cyan]export ZILLIZ_CLOUD_URI=https://in03-xxx.serverless.gcp-us-west1.cloud.zilliz.com[/cyan]"
    )
    console.print("  [cyan]export ZILLIZ_API_KEY=your_api_key_here[/cyan]")
    console.print("  [cyan]export ZILLIZ_CLUSTER_ID=your_cluster_id[/cyan]")
    console.print()
    console.print("📚 Get your credentials from: https://cloud.zilliz.com/")
    console.print("📖 Setup guide: https://docs.zilliz.com/docs/data-import")


def _test_database_connection(
    console: Console, mode: str, cloud_uri: str, api_key: str, cluster_id: str
) -> None:
    """Test database connection with provided or current settings."""
    import os
    from importlib import reload

    from . import config as config_module

    console.print("🧪 Testing database connection...")

    # Temporarily update environment if parameters provided
    if mode:
        os.environ["MILVUS_DEPLOYMENT_MODE"] = mode
    if cloud_uri:
        os.environ["ZILLIZ_CLOUD_URI"] = cloud_uri
    if api_key:
        os.environ["ZILLIZ_API_KEY"] = api_key
    if cluster_id:
        os.environ["ZILLIZ_CLUSTER_ID"] = cluster_id

    # Reload config to pick up environment changes
    reload(config_module)

    # Test connection
    from .vector_db import MilvusVectorDB

    vector_db = MilvusVectorDB()
    console.print("✅ Database connection successful!")

    deployment_mode = vector_db.config.database.deployment_mode
    if deployment_mode == "cloud":
        console.print(
            f"🌐 Connected to Zilliz Cloud: {vector_db.config.database.zilliz_cloud_uri}"
        )
    else:
        console.print(
            f"💾 Connected to local Milvus: {vector_db.config.database.milvus_uri}"
        )


@cli.command()
@click.option(
    "--cloud-uri",
    help="Zilliz Cloud endpoint URI (e.g., https://in03-xxx.serverless.gcp-us-west1.cloud.zilliz.com)",
)
@click.option(
    "--api-key",
    help="Zilliz Cloud API key",
)
@click.option(
    "--cluster-id",
    help="Zilliz Cloud cluster ID",
)
@click.option(
    "--mode",
    type=click.Choice(["local", "cloud"]),
    help="Set database deployment mode (local for Milvus Lite, cloud for Zilliz Cloud)",
)
@click.option(
    "--test-connection",
    is_flag=True,
    help="Test the database connection after configuration",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def config_db(
    cloud_uri: str,
    api_key: str,
    cluster_id: str,
    mode: str,
    test_connection: bool,
    verbose: bool,
) -> None:
    """Configure database connection (local Milvus Lite or Zilliz Cloud)."""
    console = Console()
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    try:
        _display_current_config(console)

        # Show environment variable instructions
        if mode == "cloud" or config.database.deployment_mode == "cloud":
            _show_cloud_instructions(console)

        # Test connection if requested
        if test_connection:
            _test_database_connection(console, mode, cloud_uri, api_key, cluster_id)

    except Exception as e:
        console.print(f"❌ Configuration failed: {e}")
        if "zilliz" in str(e).lower() or "cloud" in str(e).lower():
            console.print(
                "💡 Check your Zilliz Cloud credentials and network connection"
            )
        sys.exit(1)


if __name__ == "__main__":
    cli()
