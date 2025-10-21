"""Vector database operations using Milvus."""

import logging
from pathlib import Path
from typing import Optional

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from .config import config


class MilvusVectorDB:
    """Milvus vector database handler for document embeddings."""

    def __init__(self) -> None:
        """Initialize Milvus client and embedding model."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self._setup_embedding_model()
        self._setup_milvus_client()

    def _setup_embedding_model(self) -> None:
        """Set up the sentence transformer model for embeddings."""
        try:
            self.logger.info(
                f"Loading embedding model: {self.config.chunking.embedding_model}"
            )
            # Use local cache directory
            cache_folder = (
                Path(self.config.docling.artifacts_path).resolve() / "embeddings"
            )

            self.embedding_model = SentenceTransformer(
                self.config.chunking.embedding_model, cache_folder=str(cache_folder)
            )
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def _setup_milvus_client(self) -> None:
        """Set up Milvus client and ensure database directory exists."""
        # Ensure Milvus directory exists
        milvus_dir = Path(self.config.database.milvus_uri).parent
        milvus_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.logger.info(
                f"Connecting to Milvus at: {self.config.database.milvus_uri}"
            )
            self.client = MilvusClient(uri=self.config.database.milvus_uri)
            self._ensure_collection()
            self.logger.info("Milvus client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Milvus client: {e}")
            raise

    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper schema."""
        collection_name = self.config.database.collection_name

        # Check if collection exists
        if self.client.has_collection(collection_name):
            self.logger.debug(f"Collection '{collection_name}' already exists")
            return

        # Create collection with proper schema
        self.logger.info(f"Creating collection: {collection_name}")
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.config.database.embedding_dimension,
            metric_type="IP",  # Inner product for cosine similarity
            consistency_level=self.config.database.consistency_level,
            auto_id=True,  # Enable auto-generated IDs
        )
        self.logger.info(f"Collection '{collection_name}' created successfully")

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a text string."""
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
        """Insert document chunks into Milvus with embeddings."""
        try:
            collection_name = self.config.database.collection_name
            data = []

            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():  # Skip empty chunks
                    continue

                embedding = self.generate_embedding(chunk)

                # Prepare document data (no 'id' field since auto_id=True)
                doc_data = {
                    "vector": embedding,
                    "text": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                }

                # Add document-level metadata if provided
                if metadata:
                    doc_data.update(
                        {
                            "file_path": metadata.get("file_path", ""),
                            "file_size": metadata.get("file_size", 0),
                            "processed_at": metadata.get("processed_at", ""),
                            "processing_time": metadata.get("processing_time", 0.0),
                        }
                    )

                # Add chunk-level metadata (including image information)
                if chunk_metadata and i < len(chunk_metadata):
                    chunk_meta = chunk_metadata[i]
                    doc_data.update(
                        {
                            "has_images": chunk_meta.get("has_images", False),
                            "num_chunk_images": len(chunk_meta.get("images", [])),
                        }
                    )

                    # Add image hashes for this chunk
                    if chunk_meta.get("images"):
                        image_hashes = [img["hash"] for img in chunk_meta["images"]]
                        doc_data["image_hashes"] = image_hashes[:5]  # Limit to 5 hashes

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

    def search_similar(
        self, query: str, limit: int = 5, document_filter: Optional[str] = None
    ) -> list[dict]:
        """Search for similar text chunks in the database."""
        try:
            collection_name = self.config.database.collection_name
            query_embedding = self.generate_embedding(query)

            # Milvus search configuration uses IP (inner product) metric

            # Add document filter if specified
            expr = None
            if document_filter:
                expr = f"document_id == '{document_filter}'"

            # Create search parameters
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

            # Add filter if specified
            if expr:
                search_kwargs["filter"] = expr

            results = self.client.search(**search_kwargs)

            # Format results
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

                # Include image hashes if present
                if entity.get("image_hashes"):
                    formatted_result["image_hashes"] = entity["image_hashes"]

                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            self.logger.error(f"Failed to search similar chunks: {e}")
            return []

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        try:
            collection_name = self.config.database.collection_name

            if not self.client.has_collection(collection_name):
                return {"exists": False}

            # Get collection info (simplified stats)
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
        """Delete all chunks for a specific document."""
        try:
            collection_name = self.config.database.collection_name

            # Delete using expression filter
            self.client.delete(
                collection_name=collection_name, expr=f"document_id == '{doc_id}'"
            )

            self.logger.info(f"Deleted document {doc_id} from collection")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def close(self) -> None:
        """Close the Milvus client connection."""
        try:
            if hasattr(self, "client"):
                # MilvusClient doesn't have explicit close method for Lite
                self.logger.debug("Milvus client closed")
        except Exception as e:
            self.logger.error(f"Error closing Milvus client: {e}")
