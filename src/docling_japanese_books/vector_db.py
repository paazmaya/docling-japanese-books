"""Vector database operations using Milvus."""

import logging
from pathlib import Path
from typing import Optional

from pymilvus import MilvusClient
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
        """Initialize BGE-M3 embedding model with Late Chunking support."""
        try:
            self.logger.info(
                f"Loading embedding model: {self.config.chunking.embedding_model}"
            )

            self.late_chunking = LateChunkingProcessor()
            cache_folder = (
                Path(self.config.docling.artifacts_path).resolve() / "embeddings"
            )

            self.embedding_model = SentenceTransformer(
                self.config.chunking.embedding_model, cache_folder=str(cache_folder)
            )

            self.logger.info(
                "BGE-M3 embedding model loaded successfully with Late Chunking support"
            )
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def _setup_milvus_client(self) -> None:
        """Connect to Milvus Lite (local) or Zilliz Cloud based on configuration."""
        deployment_mode = self.config.database.deployment_mode

        if deployment_mode == "local":
            milvus_dir = Path(self.config.database.milvus_uri).parent
            milvus_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Using local Milvus Lite at: {self.config.database.milvus_uri}"
            )
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
            self.client = MilvusClient(**connection_params)
            self._ensure_collection()

            if deployment_mode == "cloud":
                self.logger.info("Connected to Zilliz Cloud successfully")
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
            raise

    def _ensure_collection(self) -> None:
        """Create collection if not exists with BGE-M3 schema."""
        collection_name = self.config.database.collection_name

        if self.client.has_collection(collection_name):
            self.logger.debug(f"Collection '{collection_name}' already exists")
            return
        self.logger.info(f"Creating collection: {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.config.database.embedding_dimension,
            metric_type="IP",
            consistency_level=self.config.database.consistency_level,
            auto_id=True,
        )
        self.logger.info(f"Collection '{collection_name}' created successfully")

    def generate_embedding(self, text: str) -> list[float]:
        """Convert text to BGE-M3 embedding vector."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def insert_document(
        self,
        doc_id: str,
        text_chunks: list[str],
        metadata: Optional[dict] = None,
        chunk_metadata: Optional[list[dict]] = None,
    ) -> bool:
        """Store document chunks with embeddings and metadata in Milvus."""
        try:
            collection_name = self.config.database.collection_name
            data = []

            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():
                    continue

                embedding = self.generate_embedding(chunk)
                doc_data = {
                    "vector": embedding,
                    "text": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                }

                if metadata:
                    doc_data.update(
                        {
                            "file_path": metadata.get("file_path", ""),
                            "file_size": metadata.get("file_size", 0),
                            "processed_at": metadata.get("processed_at", ""),
                            "processing_time": metadata.get("processing_time", 0.0),
                        }
                    )
                if chunk_metadata and i < len(chunk_metadata):
                    chunk_meta = chunk_metadata[i]
                    doc_data.update(
                        {
                            "has_images": chunk_meta.get("has_images", False),
                            "num_chunk_images": len(chunk_meta.get("images", [])),
                        }
                    )

                    if chunk_meta.get("images"):
                        image_hashes = [img["hash"] for img in chunk_meta["images"]]
                        doc_data["image_hashes"] = image_hashes[:5]

                data.append(doc_data)

            if data:
                result = self.client.insert(collection_name=collection_name, data=data)
                self.logger.info(
                    f"Inserted {len(data)} chunks for document {doc_id}, "
                    f"insert_count: {result.get('insert_count', 0)}"
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
        metadata: Optional[dict] = None,
        max_chunk_length: int = 800,
    ) -> bool:
        """Store document using Late Chunking for improved context preservation."""
        try:
            collection_name = self.config.database.collection_name
            chunks, chunk_embeddings = self.late_chunking.process_document(
                full_document, max_chunk_length
            )

            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                if not chunk.strip():
                    continue
                doc_data = {
                    "vector": embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding),
                    "text": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "chunking_method": "late_chunking",
                }
                if metadata:
                    doc_data.update(
                        {
                            "file_path": metadata.get("file_path", ""),
                            "file_size": metadata.get("file_size", 0),
                            "processed_at": metadata.get("processed_at", ""),
                            "processing_time": metadata.get("processing_time", 0.0),
                        }
                    )

                data.append(doc_data)

            if data:
                result = self.client.insert(collection_name=collection_name, data=data)
                self.logger.info(
                    f"Inserted {len(data)} Late Chunking chunks for document {doc_id}, "
                    f"insert_count: {result.get('insert_count', 0)}"
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
    ) -> list[dict]:
        """Find similar text chunks using embedding similarity search."""
        try:
            collection_name = self.config.database.collection_name
            query_embedding = self.generate_embedding(query)

            expr = None
            if document_filter:
                expr = f"document_id == '{document_filter}'"
            search_kwargs = {
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

            results = self.client.search(**search_kwargs)
            formatted_results = []
            for result in results[0]:
                entity = result["entity"]
                formatted_result = {
                    "text": entity["text"],
                    "document_id": entity["document_id"],
                    "chunk_index": entity["chunk_index"],
                    "file_path": entity["file_path"],
                    "similarity_score": result["distance"],
                    "has_images": entity.get("has_images", False),
                }

                if entity.get("image_hashes"):
                    formatted_result["image_hashes"] = entity["image_hashes"]

                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            self.logger.error(f"Failed to search similar chunks: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """Return basic collection information and status."""
        try:
            collection_name = self.config.database.collection_name

            if not self.client.has_collection(collection_name):
                return {"exists": False}

            stats = {
                "exists": True,
                "name": collection_name,
                "database_path": self.config.database.milvus_uri,
            }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"exists": False, "error": str(e)}

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to specified document."""
        try:
            collection_name = self.config.database.collection_name

            self.client.delete(
                collection_name=collection_name, expr=f"document_id == '{doc_id}'"
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
                self.logger.debug("Milvus client closed")
        except Exception as e:
            self.logger.error(f"Error closing Milvus client: {e}")
