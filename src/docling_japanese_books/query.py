"""Vector database query utility for Docling Japanese Books."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .vector_db import MilvusVectorDB


@click.command()
@click.argument("query", type=str)
@click.option(
    "--limit",
    "-l",
    default=5,
    help="Maximum number of results to return",
)
@click.option(
    "--document",
    "-d",
    help="Filter results to specific document ID",
)
def search(query: str, limit: int, document: str) -> None:
    """Search for similar text chunks in the vector database.

    QUERY: The text to search for similar chunks.
    """
    console = Console()

    console.print(f"\n🔍 [yellow]Searching for:[/yellow] {query}")
    if document:
        console.print(f"📄 [yellow]Filtering by document:[/yellow] {document}")
    console.print()

    try:
        # Initialize vector database
        vector_db = MilvusVectorDB()

        # Get collection stats
        stats = vector_db.get_collection_stats()
        if not stats.get("exists"):
            console.print(
                "❌ [red]Vector database collection does not exist or is empty[/red]"
            )
            console.print(
                "💡 [yellow]Run document processing first to populate the database[/yellow]"
            )
            return

        # Perform search
        results = vector_db.search_similar(
            query=query, limit=limit, document_filter=document
        )

        if not results:
            console.print("❌ [red]No similar chunks found[/red]")
            return

        # Display results in a table
        table = Table(title=f"Top {len(results)} Similar Chunks")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Document", style="green", width=18)
        table.add_column("Chunk", style="blue", width=6)
        table.add_column("Score", style="yellow", width=8)
        table.add_column("Images", style="magenta", width=6)
        table.add_column("Text Preview", style="white", width=50)

        for i, result in enumerate(results, 1):
            # Truncate text for display
            text_preview = (
                result["text"][:80] + "..."
                if len(result["text"]) > 80
                else result["text"]
            )
            text_preview = text_preview.replace("\n", " ")

            # Show image indicator
            image_indicator = "📷" if result.get("has_images") else "-"
            if result.get("image_hashes"):
                image_indicator = f"📷×{len(result['image_hashes'])}"

            table.add_row(
                str(i),
                result["document_id"],
                str(result["chunk_index"]),
                f"{result['similarity_score']:.3f}",
                image_indicator,
                text_preview,
            )

        console.print(table)

        # Show full text of top result
        if results:
            top_result = results[0]

            result_text = top_result["text"]
            title = f"Top Result: {top_result['document_id']} (Chunk {top_result['chunk_index']})"

            # Add image information to title if present
            if top_result.get("has_images") and top_result.get("image_hashes"):
                title += f" 📷 {len(top_result['image_hashes'])} image(s)"
                result_text += f"\n\n[bold]Image Hashes:[/bold] {', '.join(top_result['image_hashes'])}"

            console.print(
                Panel(
                    result_text,
                    title=title,
                    border_style="green",
                )
            )

    except Exception as e:
        console.print(f"❌ [red]Error searching database:[/red] {e}")


@click.command()
def stats() -> None:
    """Show vector database statistics."""
    console = Console()

    try:
        vector_db = MilvusVectorDB()
        stats = vector_db.get_collection_stats()

        console.print(
            Panel.fit(
                f"📊 [bold blue]Vector Database Statistics[/bold blue]\n\n"
                f"📁 Database Path: [cyan]{stats.get('database_path', 'N/A')}[/cyan]\n"
                f"📚 Collection: [green]{stats.get('name', 'N/A')}[/green]\n"
                f"✅ Exists: [yellow]{stats.get('exists', False)}[/yellow]",
                title="Milvus Database",
                border_style="blue",
            )
        )

    except Exception as e:
        console.print(f"❌ [red]Error getting database stats:[/red] {e}")


if __name__ == "__main__":
    search()
