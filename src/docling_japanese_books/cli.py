"""Command-line interface for the document processing pipeline."""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .config import config
from .downloader import ModelDownloader
from .processor import DocumentProcessor


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with rich console formatting and tracebacks."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def display_config_panel(console: Console, directory: Path) -> None:
    """Display current configuration in formatted panel."""
    console.print()
    vision_status = "Enabled" if config.docling.enable_vision else "Disabled"
    vision_display = (
        f"{vision_status} ({config.docling.vision_model})"
        if config.docling.enable_vision
        else "Disabled"
    )

    # Format database connection info based on deployment mode
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
            f"📚 Chunking Strategy: [yellow]{config.chunking.get_preferred_strategy(config.chunking.embedding_model)}[/yellow]\n"
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
    """Find supported files in directory and show type summary."""
    console.print()
    console.print("🔍 [yellow]Discovering files...[/yellow]")

    files = processor.discover_files(directory)
    if not files:
        console.print(f"⚠️ [yellow]No supported files found in {directory}[/yellow]")
        return files

    console.print(f"📄 Found {len(files)} supported files")

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
    """Show files that would be processed without executing."""
    console.print()
    console.print(
        "🔍 [yellow]Dry run - showing files that would be processed:[/yellow]"
    )
    for i, file_path in enumerate(files[:10], 1):
        console.print(f"  {i:3d}. {file_path.name}")
    if len(files) > 10:
        console.print(f"  ... and {len(files) - 10} more files")
    console.print()
    console.print("💡 Run without --dry-run to actually process the files")


def display_results(results, files: list[Path], console: Console) -> None:
    """Show processing statistics and error summary."""
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
        for error in results.errors[:5]:
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
    """Check existing models and return True to skip download if all exist."""
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
    """Show download status for each model with success/failure counts."""
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
    """Format model information for download configuration display."""
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
    """Evaluate and compare BGE-M3 Late Chunking vs traditional embedding approaches.

    For comprehensive 3-model comparison including Snowflake Arctic Embed,
    use: python scripts/evaluate_snowflake_arctic.py
    """
    import json

    console = Console()
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    try:
        # Load documents
        if documents:
            console.print(f"📖 Loading documents from: {documents}")
            with documents.open("r", encoding="utf-8") as f:
                docs = json.load(f)
        else:
            console.print("📖 Processing Japanese documents from test_docs/ folder...")
            # The evaluation will process real PDF files from test_docs/ or use realistic sample content
            docs = {}

        if docs:
            console.print(f"🔬 Evaluating {len(docs)} documents...")
        else:
            console.print("� Running evaluation with test documents...")
        console.print("�📊 Comparing Traditional vs Late Chunking approaches")
        console.print("🧠 Testing BGE-M3 multilingual model vs sentence-transformers")
        console.print()

        # Run evaluation - main() function will handle document processing
        import sys

        from .embedding_evaluation import main as run_evaluation

        original_argv = sys.argv[:]
        sys.argv = ["embedding_evaluation"]
        if output != Path("embedding_evaluation_results.json"):
            sys.argv.extend(["--output", str(output)])

        try:
            run_evaluation()
        finally:
            sys.argv = original_argv

        console.print(f"✅ Evaluation complete! Results saved to: {output}")

    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


def _display_current_config(console: Console) -> None:
    """Show active database mode and connection settings."""
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
    """Display Zilliz Cloud environment variable setup guide."""
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
    """Validate database connectivity with given or existing configuration."""
    import os
    from importlib import reload

    from . import config as config_module

    console.print("🧪 Testing database connection...")

    if mode:
        os.environ["MILVUS_DEPLOYMENT_MODE"] = mode
    if cloud_uri:
        os.environ["ZILLIZ_CLOUD_URI"] = cloud_uri
    if api_key:
        os.environ["ZILLIZ_API_KEY"] = api_key
    if cluster_id:
        os.environ["ZILLIZ_CLUSTER_ID"] = cluster_id

    reload(config_module)  # Apply environment changes
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


@cli.group()
def chunking() -> None:
    """Chunking strategy management and evaluation commands."""
    pass


@chunking.command()
@click.option(
    "--models",
    type=str,
    help="Comma-separated list of models to analyze (default: all supported models)",
)
@click.option(
    "--strategies",
    type=str,
    help="Comma-separated list of strategies to test (default: all strategies)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="chunking_analysis.json",
    help="Output file for detailed analysis results",
)
@click.option(
    "--report",
    type=click.Path(path_type=Path),
    default="chunking_report.md",
    help="Output file for human-readable report",
)
@click.option(
    "--quick", is_flag=True, help="Quick analysis with minimal models for testing"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def analyze(
    models: str, strategies: str, output: Path, report: Path, quick: bool, verbose: bool
) -> None:
    """
    Comprehensive analysis of chunking strategies across embedding models.

    This command evaluates all available chunking strategies (late, traditional,
    hybrid, hierarchical) with all supported embedding models, documenting their
    capabilities, limitations, and performance characteristics.

    Results include:
    - Model-strategy compatibility matrix
    - Performance benchmarks and recommendations
    - Fallback mechanisms and alternatives
    - Japanese-specific evaluation metrics
    - Production deployment recommendations

    Examples:

        # Analyze all models and strategies
        docling-japanese-books chunking analyze

        # Quick test with minimal models
        docling-japanese-books chunking analyze --quick

        # Test specific models
        docling-japanese-books chunking analyze --models "BAAI/bge-m3,jinaai/jina-embeddings-v4"

        # Analyze with custom output files
        docling-japanese-books chunking analyze --output results.json --report analysis.md
    """
    console = Console()
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    console.print("🔍 [bold blue]Chunking Strategy Analysis[/bold blue]")
    console.print()

    # Parse model list
    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",")]
        console.print(f"📋 Testing models: {', '.join(model_list)}")
    elif quick:
        model_list = ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]
        console.print("⚡ Quick mode: Testing minimal model set")
    else:
        console.print("📋 Testing all available models")

    # Parse strategy list
    if strategies:
        strategy_list = [s.strip() for s in strategies.split(",")]
        console.print(f"⚙️  Testing strategies: {', '.join(strategy_list)}")
    else:
        console.print("⚙️  Testing all available strategies")

    console.print(f"💾 Results will be saved to: {output}")
    console.print(f"📄 Report will be saved to: {report}")
    console.print()

    try:
        # Import and run the analysis
        import sys
        from pathlib import Path as ImportPath

        # Add scripts directory to path
        scripts_dir = ImportPath(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))

        from evaluate_all_chunking_strategies import ChunkingStrategyAnalyzer

        analyzer = ChunkingStrategyAnalyzer()

        console.print("🔄 Starting comprehensive analysis...")
        with console.status("[bold green]Analyzing chunking strategies..."):
            analysis = analyzer.analyze_model_capabilities(model_list)

        # Save results
        import json

        with open(output, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"✅ Analysis results saved to: {output}")

        # Generate report
        analyzer.generate_report(analysis, str(report))
        console.print(f"✅ Human-readable report saved to: {report}")

        # Display summary
        console.print()
        console.print("📊 [bold]Analysis Summary[/bold]")
        console.print()

        for model_name, model_data in analysis["models"].items():
            supported = model_data.get("supported_strategies", [])
            failed = model_data.get("failed_strategies", [])

            console.print(f"🤖 [yellow]{model_name}[/yellow]")
            console.print(
                f"   ✅ Supported: {', '.join(supported) if supported else 'None'}"
            )
            console.print(f"   ❌ Failed: {', '.join(failed) if failed else 'None'}")

            if model_data.get("best_strategy"):
                best = model_data["best_strategy"]
                console.print(
                    f"   🏆 Best: {best['strategy']} (context: {best['metrics']['context_preservation_score']:.3f})"
                )
            console.print()

        # Display recommendations
        console.print("💡 [bold]Key Recommendations[/bold]")
        for use_case, recs in analysis.get("recommendations", {}).items():
            console.print(f"🎯 [cyan]{use_case.replace('_', ' ').title()}[/cyan]:")
            for rec in recs[:2]:  # Show first 2 recommendations
                console.print(f"   • {rec}")
            console.print()

    except Exception as e:
        console.print(f"❌ Analysis failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@chunking.command()
@click.option(
    "--model",
    type=str,
    default=None,
    help="Specific model to show capabilities for (default: current configured model)",
)
def capabilities(model: str) -> None:
    """
    Show chunking strategy capabilities for embedding models.

    Displays which chunking strategies are supported by each model,
    their performance characteristics, and recommended use cases.

    Examples:

        # Show capabilities for current model
        docling-japanese-books chunking capabilities

        # Show capabilities for specific model
        docling-japanese-books chunking capabilities --model "BAAI/bge-m3"
    """
    console = Console()

    # Import the evaluator to get model configs
    from .embedding_evaluation import MultiStrategyEmbeddingEvaluator

    evaluator = MultiStrategyEmbeddingEvaluator()

    if model is None:
        model = config.chunking.embedding_model
        console.print(
            f"📋 Showing capabilities for current model: [yellow]{model}[/yellow]"
        )
    else:
        console.print(f"📋 Showing capabilities for: [yellow]{model}[/yellow]")

    console.print()

    # Get model configuration
    model_config = evaluator.model_configs.get(model, {})
    if not model_config:
        console.print(f"❌ Model [red]{model}[/red] not found in supported models")
        console.print("📚 Supported models:")
        for supported_model in evaluator.model_configs.keys():
            console.print(f"   • {supported_model}")
        return

    # Display capabilities
    console.print(
        Panel.fit(
            f"🤖 [bold]{model}[/bold]\n\n"
            f"📝 Notes: {model_config.get('notes', 'No notes available')}\n"
            f"🎯 Task: {model_config.get('task', 'None')}\n"
            f"📏 Optimal Chunk Size: {model_config.get('optimal_chunk_size', 'Unknown')} chars\n"
            f"🔗 Late Chunking Support: {'✅ Yes' if model_config.get('supports_late_chunking') else '❌ No'}",
            title="Model Information",
        )
    )

    # Show supported strategies
    supported_strategies = model_config.get("supported_strategies", [])
    console.print("⚙️  [bold]Supported Chunking Strategies[/bold]")
    console.print()

    for strategy in supported_strategies:
        strategy_info = evaluator.strategy_definitions.get(strategy, {})
        console.print(
            f"✅ [green]{strategy}[/green] - {strategy_info.get('name', strategy)}"
        )
        console.print(
            f"   {strategy_info.get('description', 'No description available')}"
        )

        best_for = strategy_info.get("best_for", [])
        if best_for:
            console.print(f"   🎯 Best for: {', '.join(best_for)}")
        console.print()

    # Show recommendations
    console.print("💡 [bold]Recommendations[/bold]")
    console.print()

    preferred = model_config.get("preferred_strategy", "traditional")
    fallbacks = model_config.get("fallback_strategies", [])

    console.print(f"🏆 Preferred Strategy: [green]{preferred}[/green]")
    if fallbacks:
        console.print(f"🔄 Fallback Options: {', '.join(fallbacks)}")

    # Usage examples
    console.print()
    console.print("🔧 [bold]Usage Examples[/bold]")
    console.print()
    console.print("# Use with this model in your code:")
    console.print(
        "from docling_japanese_books.enhanced_chunking import create_chunking_strategy"
    )
    console.print(f"chunker = create_chunking_strategy('{model}', '{preferred}')")
    if model_config.get("task"):
        console.print("# Or with task specification:")
        console.print(
            f"chunker = create_chunking_strategy('{model}', '{preferred}', task='{model_config['task']}')"
        )


@chunking.command()
@click.argument(
    "strategy",
    type=click.Choice(["auto", "late", "traditional", "hybrid", "hierarchical"]),
)
@click.option(
    "--model", type=str, help="Set embedding model along with strategy (optional)"
)
def set_strategy(strategy: str, model: str) -> None:
    """
    Set the default chunking strategy for document processing.

    STRATEGY options:
    - auto: Automatically select best strategy per model
    - late: Late chunking (embed full document, then chunk)
    - traditional: Traditional chunking (chunk first, then embed)
    - hybrid: Model-adaptive strategy with fallbacks
    - hierarchical: Multiple chunk sizes for different queries

    Examples:

        # Set to auto-select best strategy per model
        docling-japanese-books chunking set-strategy auto

        # Force late chunking for all models (will fallback if not supported)
        docling-japanese-books chunking set-strategy late

        # Set strategy and model together
        docling-japanese-books chunking set-strategy hybrid --model "jinaai/jina-embeddings-v4"
    """
    console = Console()

    console.print(f"⚙️  Setting chunking strategy to: [yellow]{strategy}[/yellow]")

    if model:
        console.print(f"🤖 Setting embedding model to: [yellow]{model}[/yellow]")

        # Validate model exists
        from .embedding_evaluation import MultiStrategyEmbeddingEvaluator

        evaluator = MultiStrategyEmbeddingEvaluator()

        if model not in evaluator.model_configs:
            console.print(
                f"⚠️  [yellow]Warning:[/yellow] Model [red]{model}[/red] not in supported models list"
            )
            console.print("📚 Supported models:")
            for supported_model in evaluator.model_configs.keys():
                console.print(f"   • {supported_model}")
            console.print()
            console.print(
                "Proceeding anyway - model may still work with fallback strategies..."
            )

    # Update configuration (this is a demo - in practice you'd update config files)
    console.print()
    console.print("📝 [bold]Configuration Update[/bold]")
    console.print(
        f"Current strategy: [bright_black]{config.chunking.chunking_strategy}[/bright_black]"
    )
    console.print(
        f"Current model: [bright_black]{config.chunking.embedding_model}[/bright_black]"
    )
    console.print()
    console.print(f"New strategy: [green]{strategy}[/green]")
    if model:
        console.print(f"New model: [green]{model}[/green]")

    console.print()
    console.print(
        "⚠️  [yellow]Note:[/yellow] This is a demonstration. To persist changes, update your config.py file:"
    )
    console.print()
    console.print(f"chunking_strategy = '{strategy}'")
    if model:
        console.print(f"embedding_model = '{model}'")


if __name__ == "__main__":
    cli()
