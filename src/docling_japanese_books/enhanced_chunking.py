"""
Enhanced chunking strategies supporting multiple embedding models.

This module extends the current late chunking implementation to support
more models and provides hybrid chunking strategies for better performance.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def process_document(
        self, document: str, max_chunk_length: int = 500
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Process document and return chunks with embeddings."""
        pass


class UniversalLateChunking(ChunkingStrategy):
    """Late chunking implementation that works with any sentence-transformer model."""

    def __init__(self, model_name: str, task: Optional[str] = None):
        """Initialize with specific model.

        Args:
            model_name: HuggingFace model identifier
            task: Task type for models that support it (e.g., Jina v4)
        """
        self.model_name = model_name
        self.task = task
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load model with appropriate configuration."""
        if self.model is not None:
            return

        logger.info(f"Loading model for late chunking: {self.model_name}")

        # Special handling for different model types
        if "jina-embeddings-v4" in self.model_name:
            # Jina v4 with task specification
            model_kwargs = {}
            if self.task:
                model_kwargs["default_task"] = self.task

            self.model = SentenceTransformer(
                self.model_name, trust_remote_code=True, model_kwargs=model_kwargs
            )
        else:
            # Standard sentence-transformers models
            self.model = SentenceTransformer(self.model_name)

        # Try to get access to underlying transformer for token embeddings
        try:
            if hasattr(self.model, "_modules") and "0" in self.model._modules:
                transformer_module = self.model._modules["0"]
                if hasattr(transformer_module, "auto_model"):
                    self.tokenizer = transformer_module.tokenizer
                    self.underlying_model = transformer_module.auto_model
                    self._supports_token_level = True
                    logger.info("Token-level embeddings available")
                else:
                    self._supports_token_level = False
                    logger.warning(
                        "Token-level embeddings not available, using sentence-level fallback"
                    )
            else:
                self._supports_token_level = False
                logger.warning(
                    "Token-level embeddings not available, using sentence-level fallback"
                )
        except Exception as e:
            logger.warning(f"Could not access token-level embeddings: {e}")
            self._supports_token_level = False

    def get_token_embeddings(self, document: str) -> torch.Tensor:
        """Extract token-level embeddings from document."""
        if not self._supports_token_level:
            # Fallback to sentence-level embedding repeated for compatibility
            sentence_embedding = self.model.encode([document], convert_to_tensor=True)
            return sentence_embedding.unsqueeze(1)  # Add sequence dimension

        # True token-level extraction
        inputs = self.tokenizer(
            document,
            return_tensors="pt",
            max_length=8192,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.underlying_model(**inputs)
            token_embeddings = outputs.last_hidden_state

        return token_embeddings

    def chunk_document(
        self, document: str, max_chunk_length: int = 500
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Enhanced chunking with better Japanese text handling."""
        import re

        # Japanese sentence endings with more comprehensive patterns
        sentence_patterns = [
            r"[。！？]+",  # Standard endings
            r"[\.!?]+",  # Western punctuation
            r"」[。！？]*",  # Quote endings
            r"』[。！？]*",  # Book quote endings
        ]

        chunks = []
        span_annotations = []
        current_chunk = ""
        chunk_start = 0
        char_pos = 0

        # Split by any of the sentence patterns
        combined_pattern = "|".join(f"({pattern})" for pattern in sentence_patterns)
        sentences = re.split(combined_pattern, document)

        for sentence in sentences:
            if not sentence or sentence.strip() in [
                "",
                "。",
                "！",
                "？",
                ".",
                "!",
                "?",
                "」",
                "』",
            ]:
                continue

            sentence = sentence.strip()

            # Check if adding this sentence exceeds chunk limit
            if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                span_annotations.append((chunk_start, char_pos))

                # Start new chunk
                current_chunk = sentence
                chunk_start = char_pos
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

            char_pos += len(sentence) + 1  # +1 for space

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            span_annotations.append((chunk_start, len(document)))

        return chunks, span_annotations

    def late_chunk_embeddings(
        self,
        token_embeddings: torch.Tensor,
        span_annotations: List[Tuple[int, int]],
        document: str,
        pooling: str = "mean",
    ) -> List[np.ndarray]:
        """Apply late chunking to token embeddings."""
        if not self._supports_token_level:
            # Fallback: return sentence embedding for each chunk
            sentence_embedding = token_embeddings[0, 0].cpu().numpy()
            return [sentence_embedding for _ in span_annotations]

        pooled_embeddings = []
        seq_len = token_embeddings.shape[1]
        doc_length = len(document)

        for char_start, char_end in span_annotations:
            # Map character positions to token positions (approximate)
            token_start = max(0, int((char_start / doc_length) * seq_len))
            token_end = min(
                seq_len, max(token_start + 1, int((char_end / doc_length) * seq_len))
            )

            # Extract chunk tokens and pool
            chunk_tokens = token_embeddings[0, token_start:token_end]

            if pooling == "mean":
                chunk_embedding = chunk_tokens.mean(dim=0)
            elif pooling == "max":
                chunk_embedding = chunk_tokens.max(dim=0)[0]
            elif pooling == "cls":
                # Use first token (usually CLS token)
                chunk_embedding = chunk_tokens[0]
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling}")

            pooled_embeddings.append(chunk_embedding.detach().cpu().numpy())

        return pooled_embeddings

    def process_document(
        self, document: str, max_chunk_length: int = 500
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Process document with enhanced late chunking."""
        self.load_model()

        # Step 1: Chunk the document
        chunks, span_annotations = self.chunk_document(document, max_chunk_length)

        # Step 2: Get token embeddings for full document
        token_embeddings = self.get_token_embeddings(document)

        # Step 3: Apply late chunking
        chunk_embeddings = self.late_chunk_embeddings(
            token_embeddings, span_annotations, document
        )

        logger.info(
            f"Processed document into {len(chunks)} chunks using enhanced late chunking"
        )
        return chunks, chunk_embeddings


class HybridChunkingStrategy(ChunkingStrategy):
    """Hybrid strategy combining multiple chunking approaches."""

    def __init__(self, model_name: str, strategies: List[str] = None):
        """Initialize hybrid chunking.

        Args:
            model_name: Model to use for embeddings
            strategies: List of strategies to combine ["late", "traditional", "hierarchical"]
        """
        self.model_name = model_name
        self.strategies = strategies or ["late", "traditional"]
        self.late_chunker = UniversalLateChunking(model_name)
        self.model = None

    def load_model(self):
        """Load model for traditional chunking."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def traditional_chunking(
        self, document: str, max_chunk_length: int = 500
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Traditional chunk-first approach."""
        chunks, _ = self.late_chunker.chunk_document(document, max_chunk_length)
        embeddings = self.model.encode(chunks)
        return chunks, list(embeddings)

    def hierarchical_chunking(
        self, document: str
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Multi-level chunking for different query types."""
        # Small chunks for detailed queries
        small_chunks, small_embeddings = self.process_chunks(document, max_length=200)

        # Medium chunks for balanced queries
        medium_chunks, medium_embeddings = self.process_chunks(document, max_length=500)

        # Large chunks for contextual queries
        large_chunks, large_embeddings = self.process_chunks(document, max_length=1000)

        # Combine all chunks with metadata about their size
        all_chunks = []
        all_embeddings = []

        for i, (chunk, emb) in enumerate(zip(small_chunks, small_embeddings)):
            all_chunks.append(f"[SMALL-{i}] {chunk}")
            all_embeddings.append(emb)

        for i, (chunk, emb) in enumerate(zip(medium_chunks, medium_embeddings)):
            all_chunks.append(f"[MEDIUM-{i}] {chunk}")
            all_embeddings.append(emb)

        for i, (chunk, emb) in enumerate(zip(large_chunks, large_embeddings)):
            all_chunks.append(f"[LARGE-{i}] {chunk}")
            all_embeddings.append(emb)

        return all_chunks, all_embeddings

    def process_chunks(
        self, document: str, max_length: int
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Process document with specific chunk size."""
        if "late" in self.strategies:
            return self.late_chunker.process_document(document, max_length)
        else:
            return self.traditional_chunking(document, max_length)

    def process_document(
        self, document: str, max_chunk_length: int = 500
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Process document with hybrid approach."""
        self.load_model()

        if "hierarchical" in self.strategies:
            return self.hierarchical_chunking(document)
        elif "late" in self.strategies:
            return self.late_chunker.process_document(document, max_chunk_length)
        else:
            return self.traditional_chunking(document, max_chunk_length)


# Factory function for easy strategy selection
def create_chunking_strategy(
    model_name: str, strategy: str = "late", task: Optional[str] = None
) -> ChunkingStrategy:
    """Create appropriate chunking strategy.

    Args:
        model_name: Model identifier
        strategy: "late", "traditional", "hybrid", "hierarchical"
        task: Task for models that support it

    Returns:
        Chunking strategy instance
    """
    if strategy == "late":
        return UniversalLateChunking(model_name, task)
    elif strategy == "hybrid":
        return HybridChunkingStrategy(model_name)
    elif strategy == "hierarchical":
        return HybridChunkingStrategy(model_name, ["hierarchical", "late"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
