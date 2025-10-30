"""
Vector database operations using Milvus.

Milvus Lite docs: https://milvus.io/docs/milvus_lite.md
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pymilvus import MilvusClient  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer

from .config import config
from .late_chunking import LateChunkingProcessor


class MilvusVectorDB:
    """Milvus vector database handler for document embeddings."""

    def __init__(self) -> None:
        """Initialize Milvus client and embedding model."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self._setup_embedding_model()
        self._setup_milvus_client()

    def _setup_embedding_model(self) -> None:
        """
        Initialize embedding model with adaptive chunking strategy support.

        This method automatically selects the best chunking strategy for the configured
        embedding model based on its capabilities and performance characteristics.

        Model-Strategy Selection Logic:
        ==============================
        - BGE-M3: Prefers late chunking for better context preservation
        - Jina v4: Uses hybrid chunking with task-specific optimization
        - Snowflake Arctic: Uses traditional chunking optimized for speed
        - Others: Falls back to traditional chunking

        Fallback Mechanisms:
        ===================
        If the preferred strategy fails, the system automatically falls back to
        increasingly basic strategies: hybrid → traditional → error
        """
        try:
            model_name = self.config.chunking.embedding_model
            self.logger.info(f"Loading embedding model: {model_name}")

            # Determine and initialize chunking strategy
            preferred_strategy, fallback_strategies = (
                self._determine_chunking_strategies(model_name)
            )
            self.chunking_strategy, self.strategy_used = (
                self._initialize_chunking_strategy(
                    model_name, preferred_strategy, fallback_strategies
                )
            )

            # Keep legacy late_chunking for backward compatibility
            self.late_chunking = LateChunkingProcessor()

            # Load sentence transformer model
            self._load_sentence_transformer(model_name)

        except Exception as e:
            self.logger.error(f"Failed to setup embedding model: {e}")
            raise

    def _determine_chunking_strategies(self, model_name: str) -> tuple[str, list[str]]:
        """Determine preferred and fallback chunking strategies for a model."""
        model_lower = model_name.lower()

        if "bge-m3" in model_lower:
            self.logger.info(
                "Using late chunking for BGE-M3 (optimal for Japanese context)"
            )
            return "late", ["hybrid", "traditional"]
        elif "jina-embeddings-v4" in model_lower:
            self.logger.info("Using hybrid chunking for Jina v4 (quantization-aware)")
            return "hybrid", ["traditional"]
        elif "snowflake" in model_lower:
            self.logger.info(
                "Using traditional chunking for Snowflake Arctic (speed-optimized)"
            )
            return "traditional", ["hybrid"]
        else:
            self.logger.info(
                f"Using traditional chunking for {model_name} (safe default)"
            )
            return "traditional", ["hybrid"]

    def _initialize_chunking_strategy(
        self, model_name: str, preferred_strategy: str, fallback_strategies: list[str]
    ) -> tuple[Any, str]:  # type: ignore[misc]
        """Initialize chunking strategy with fallback handling."""
        from .enhanced_chunking import create_chunking_strategy

        strategies_to_try = [preferred_strategy] + fallback_strategies

        for strategy in strategies_to_try:
            try:
                # Determine task for task-aware models
                task = (
                    "retrieval" if "jina-embeddings-v4" in model_name.lower() else None
                )

                chunking_strategy = create_chunking_strategy(model_name, strategy, task)

                if strategy != preferred_strategy:
                    self.logger.warning(
                        f"Preferred strategy '{preferred_strategy}' failed, "
                        f"using fallback '{strategy}'"
                    )
                else:
                    self.logger.info(
                        f"Successfully initialized {strategy} chunking strategy"
                    )

                return chunking_strategy, strategy

            except Exception as e:
                self.logger.warning(
                    f"Strategy '{strategy}' failed for {model_name}: {e}"
                )
                continue

        raise RuntimeError(f"All chunking strategies failed for {model_name}")

    def _load_sentence_transformer(self, model_name: str) -> None:
        """Load SentenceTransformer model with appropriate configuration."""
        cache_folder = Path(self.config.docling.artifacts_path).resolve() / "embeddings"

        try:
            if "jina-embeddings-v4" in model_name.lower():
                self.embedding_model = SentenceTransformer(
                    model_name,
                    cache_folder=str(cache_folder),
                    trust_remote_code=True,
                    model_kwargs={"default_task": "retrieval"},
                )
            else:
                self.embedding_model = SentenceTransformer(
                    model_name, cache_folder=str(cache_folder)
                )

            self.logger.info(
                f"Embedding model loaded successfully with {self.strategy_used} chunking strategy"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load SentenceTransformer directly: {e}")
            self.embedding_model = None
            self.logger.info("Will use chunking strategy for all operations")

    def _setup_milvus_client(self) -> None:
        """Connect to Milvus (Lite, Docker, or Zilliz Cloud) based on configuration."""
        deployment_mode = self.config.database.deployment_mode

        if deployment_mode == "local":
            milvus_dir = Path(self.config.database.milvus_uri).parent
            milvus_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Using local Milvus Lite at: {self.config.database.milvus_uri}"
            )
        elif deployment_mode == "docker":
            self.logger.info(
                f"Connecting to Milvus Docker at: {self.config.database.milvus_uri}"
            )
            # No special setup needed, just connect to the Docker host/port
        elif deployment_mode == "cloud":
            self.logger.info(
                f"Connecting to Zilliz Cloud at: {self.config.database.zilliz_cloud_uri}"
            )
            if self.config.database.zilliz_cluster_id:
                self.logger.info(
                    f"Cluster ID: {self.config.database.zilliz_cluster_id}"
                )
        else:
            raise ValueError(f"Unsupported deployment mode: {deployment_mode}")

        try:
            connection_params = self.config.database.get_connection_params()
            self.logger.info(f"Connecting to Milvus ({deployment_mode} mode)...")
            self.client = MilvusClient(**connection_params)  # type: ignore[misc]
            self._ensure_collection()

            if deployment_mode == "cloud":
                self.logger.info("Connected to Zilliz Cloud successfully")
            elif deployment_mode == "docker":
                self.logger.info("Connected to Milvus Docker successfully")
            else:
                self.logger.info("Connected to local Milvus Lite successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize Milvus client ({deployment_mode} mode): {e}"
            )
            if deployment_mode == "cloud":
                self.logger.error(
                    "Check your Zilliz Cloud URI and API key configuration"
                )
            elif deployment_mode == "docker":
                self.logger.error("Check your Docker container and port mapping")
            raise

    def _ensure_collection(self) -> None:
        """Create collection if not exists with BGE-M3 schema."""
        collection_name = self.config.database.collection_name

        if self.client.has_collection(collection_name):  # type: ignore[misc]
            self.logger.debug(f"Collection '{collection_name}' already exists")
            return
        self.logger.info(f"Creating collection: {collection_name}")
        self.client.create_collection(  # type: ignore[misc]
            collection_name=collection_name,
            dimension=self.config.chunking.embedding_dim,  # type: ignore[misc]
        )
        self.logger.info(f"Collection '{collection_name}' created successfully")

    def generate_embedding(self, text: str) -> list[float]:
        """Convert text to BGE-M3 embedding vector."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)  # type: ignore[misc]
            return embedding.tolist()  # type: ignore[misc]
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def insert_document(
        self,
        doc_id: str,
        text_chunks: list[str],
        metadata: Optional[dict[str, Any]] = None,
        chunk_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        """Store document chunks with embeddings and metadata in Milvus."""
        try:
            collection_name = self.config.database.collection_name
            data = []

            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():
                    continue

                embedding = self.generate_embedding(chunk)
                doc_data = {  # type: ignore[misc]
                    "vector": embedding,
                    "text": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                }

                if metadata:
                    doc_data.update(  # type: ignore[misc]
                        {
                            "file_path": metadata.get("file_path", ""),  # type: ignore[misc]
                            "file_size": metadata.get("file_size", 0),  # type: ignore[misc]
                            "processed_at": metadata.get("processed_at", ""),  # type: ignore[misc]
                            "processing_time": metadata.get("processing_time", 0.0),  # type: ignore[misc]
                        }
                    )
                if chunk_metadata and i < len(chunk_metadata):  # type: ignore[misc]
                    chunk_meta = chunk_metadata[i]  # type: ignore[misc]
                    doc_data.update(  # type: ignore[misc]
                        {
                            "has_images": chunk_meta.get("has_images", False),  # type: ignore[misc]
                            "num_chunk_images": len(chunk_meta.get("images", [])),  # type: ignore[misc]
                        }
                    )

                    if chunk_meta.get("images"):  # type: ignore[misc]
                        image_hashes = [img["hash"] for img in chunk_meta["images"]]  # type: ignore[misc]
                        doc_data["image_hashes"] = image_hashes[:5]  # type: ignore[misc]

                data.append(doc_data)  # type: ignore[misc]

            if data:
                result = self.client.insert(collection_name=collection_name, data=data)  # type: ignore[misc]
                self.logger.info(
                    f"Inserted {len(data)} chunks for document {doc_id}, "  # type: ignore[misc]
                    f"insert_count: {result.get('insert_count', 0)}"  # type: ignore[misc]
                )
                return True
            else:
                self.logger.warning(f"No valid chunks to insert for document {doc_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to insert document {doc_id}: {e}")
            return False

    def insert_document_with_late_chunking(
        self,
        doc_id: str,
        full_document: str,
        metadata: Optional[dict[str, Any]] = None,
        max_chunk_length: int = 800,
    ) -> bool:
        """Store document using Late Chunking for improved context preservation."""
        try:
            collection_name = self.config.database.collection_name
            chunks, chunk_embeddings = self.late_chunking.process_document(  # type: ignore[misc]
                full_document, max_chunk_length
            )

            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):  # type: ignore[misc]
                if not chunk.strip():
                    continue
                doc_data = {  # type: ignore[misc]
                    "vector": embedding.tolist()  # type: ignore[misc]
                    if hasattr(embedding, "tolist")  # type: ignore[misc]
                    else list(embedding),  # type: ignore[misc]
                    "text": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "chunking_method": "late_chunking",
                }
                if metadata:
                    doc_data.update(  # type: ignore[misc]
                        {
                            "file_path": metadata.get("file_path", ""),  # type: ignore[misc]
                            "file_size": metadata.get("file_size", 0),  # type: ignore[misc]
                            "processed_at": metadata.get("processed_at", ""),  # type: ignore[misc]
                            "processing_time": metadata.get("processing_time", 0.0),  # type: ignore[misc]
                        }
                    )

                data.append(doc_data)  # type: ignore[misc]

            if data:
                result = self.client.insert(collection_name=collection_name, data=data)  # type: ignore[misc]
                self.logger.info(
                    f"Inserted {len(data)} Late Chunking chunks for document {doc_id}, "  # type: ignore[misc]
                    f"insert_count: {result.get('insert_count', 0)}"  # type: ignore[misc]
                )
                return True
            else:
                self.logger.warning(f"No valid chunks to insert for document {doc_id}")
                return False

        except Exception as e:
            self.logger.error(
                f"Failed to insert document with Late Chunking {doc_id}: {e}"
            )
            return False

    def search_similar(
        self, query: str, limit: int = 5, document_filter: Optional[str] = None
    ) -> list[dict[str, Any]]:  # type: ignore[misc]
        """Find similar text chunks using embedding similarity search."""
        try:
            collection_name = self.config.database.collection_name
            query_embedding = self.generate_embedding(query)

            expr = None
            if document_filter:
                expr = f"document_id == '{document_filter}'"
            search_kwargs = {  # type: ignore[misc]
                "collection_name": collection_name,
                "data": [query_embedding],
                "limit": limit,
                "output_fields": [
                    "text",
                    "document_id",
                    "chunk_index",
                    "file_path",
                    "has_images",
                    "image_hashes",
                ],
            }

            if expr:
                search_kwargs["filter"] = expr

            results = self.client.search(**search_kwargs)  # type: ignore[misc]
            formatted_results = []
            for result in results[0]:  # type: ignore[misc]
                entity = result["entity"]  # type: ignore[misc]
                formatted_result = {  # type: ignore[misc]
                    "text": entity["text"],  # type: ignore[misc]
                    "document_id": entity["document_id"],  # type: ignore[misc]
                    "chunk_index": entity["chunk_index"],  # type: ignore[misc]
                    "file_path": entity["file_path"],  # type: ignore[misc]
                    "similarity_score": result["distance"],  # type: ignore[misc]
                    "has_images": entity.get("has_images", False),  # type: ignore[misc]
                }

                if entity.get("image_hashes"):  # type: ignore[misc]
                    formatted_result["image_hashes"] = entity["image_hashes"]  # type: ignore[misc]

                formatted_results.append(formatted_result)  # type: ignore[misc]

            return formatted_results  # type: ignore[misc]

        except Exception as e:
            self.logger.error(f"Failed to search similar chunks: {e}")
            return []  # type: ignore[misc]

    def get_collection_stats(self) -> dict[str, Any]:  # type: ignore[misc]
        """Return basic collection information and status."""
        try:
            collection_name = self.config.database.collection_name

            if not self.client.has_collection(collection_name):  # type: ignore[misc]
                return {"exists": False}  # type: ignore[misc]

            stats = {  # type: ignore[misc]
                "exists": True,
                "name": collection_name,
                "database_path": self.config.database.milvus_uri,
            }

            return stats  # type: ignore[misc]

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"exists": False, "error": str(e)}  # type: ignore[misc]

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to specified document."""
        try:
            collection_name = self.config.database.collection_name

            self.client.delete(  # type: ignore[misc]
                collection_name=collection_name,
                filter=f'document_id == "{doc_id}"',
            )

            self.logger.info(f"Deleted document {doc_id} from collection")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def close(self) -> None:
        """Clean up Milvus client resources."""
        try:
            if hasattr(self, "client"):
                self.client.close()
                self.logger.debug("Milvus client closed")
        except Exception as e:
            self.logger.error(f"Error closing Milvus client: {e}")
