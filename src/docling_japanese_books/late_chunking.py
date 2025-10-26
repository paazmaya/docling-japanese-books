"""
Late Chunking implementation for improved context preservation.

Based on Milvus blog post: "Smarter Retrieval for RAG: Late Chunking with Jina Embeddings v2"
https://milvus.io/blog/smarter-retrieval-for-rag-late-chunking-with-jina-embeddings-v2-and-milvus.md

Late Chunking flips the traditional embedding pipeline:
1. Embed first: Generate token embeddings for the full document with global context
2. Chunk later: Average-pool contiguous token spans to form chunk vectors

This preserves cross-chunk context and improves retrieval accuracy, especially
important for Japanese text with complex grammatical structures.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import config

# Suppress FlagEmbedding's custom warnings
warnings.filterwarnings("ignore", module="FlagEmbedding")

logger = logging.getLogger(__name__)


class LateChunkingProcessor:
    """Implements Late Chunking for improved context-aware embeddings."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the Late Chunking processor.

        Args:
            model_name: Model to use for embeddings. Defaults to config setting.
        """
        self.model_name = model_name or config.chunking.embedding_model
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the embedding model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            # Model: https://huggingface.co/BAAI/bge-m3

            # BGE-M3 uses FlagEmbedding under the hood
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info("Using optimized BGE-M3 implementation")
                self.model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=torch.cuda.is_available(),
                    device=self.device,
                )
                self._use_flag_model = True

            except ImportError:
                logger.info("FlagEmbedding not available, using transformers")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True
                ).to(self.device)
                self._use_flag_model = False

    def simple_sentence_chunker(
        self, document: str, max_chunk_length: int = 500
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """Simple sentence-based chunking with character-level annotations.

        Args:
            document: Input document text
            max_chunk_length: Maximum characters per chunk

        Returns:
            Tuple of (chunks, span_annotations) where span_annotations are
            character-level start/end positions
        """
        # Simple sentence splitting for Japanese text
        import re

        # Japanese sentence endings: 。！？
        sentence_endings = re.compile(r"[。！？]+")
        sentences = sentence_endings.split(document)

        chunks = []
        span_annotations = []
        current_chunk = ""
        chunk_start = 0
        char_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find the actual sentence ending in original text
            sentence_end_match = sentence_endings.search(document, char_pos)
            if sentence_end_match:
                full_sentence = document[char_pos : sentence_end_match.end()].strip()
                next_char_pos = sentence_end_match.end()
            else:
                full_sentence = sentence
                next_char_pos = char_pos + len(sentence)

            # Check if adding this sentence exceeds chunk limit
            if (
                len(current_chunk) + len(full_sentence) > max_chunk_length
                and current_chunk
            ):
                # Finalize current chunk
                chunk_end = char_pos
                chunks.append(current_chunk.strip())
                span_annotations.append((chunk_start, chunk_end))

                # Start new chunk
                current_chunk = full_sentence
                chunk_start = char_pos
            else:
                current_chunk += full_sentence

            char_pos = next_char_pos

        # Add the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            span_annotations.append((chunk_start, len(document)))

        return chunks, span_annotations

    def document_to_token_embeddings(
        self, document: str, batch_size: int = 4096
    ) -> torch.Tensor:
        """Generate token-level embeddings for the entire document.

        Args:
            document: Input document text
            batch_size: Token batch size for processing

        Returns:
            Token embeddings tensor [1, seq_len, hidden_size]
        """
        self.load_model()

        if self._use_flag_model:
            # BGE-M3 has built-in tokenization
            # For late chunking, we need to access token-level embeddings
            # This is a simplified approach - for production, you'd want
            # to access the model's internal representations
            logger.warning("BGE-M3 FlagModel doesn't expose token embeddings directly")
            logger.info("Falling back to sentence-level embeddings")

            # Get dense embedding (this is sentence-level, not token-level)
            embedding = self.model.encode(
                [document],
                batch_size=1,
                max_length=8192,  # BGE-M3 supports 8K context
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"]

            # Reshape to match expected format [batch, seq_len, hidden_size]
            # Note: This is a fallback - true late chunking requires token embeddings
            return torch.tensor(embedding, device=self.device).unsqueeze(1)

        else:
            # Use transformers model for true token-level embeddings
            tokenized_document = self.tokenizer(
                document,
                return_tensors="pt",
                max_length=8192,
                truncation=True,
                padding=True,
            ).to(self.device)

            outputs = []
            tokens = tokenized_document["input_ids"][0]

            # Process in batches to handle long documents
            for i in range(0, len(tokens), batch_size):
                start = i
                end = min(i + batch_size, len(tokens))

                batch_inputs = {
                    k: v[:, start:end] for k, v in tokenized_document.items()
                }

                with torch.no_grad():
                    model_output = self.model(**batch_inputs)
                    outputs.append(model_output.last_hidden_state)

            return torch.cat(outputs, dim=1)

    def late_chunking(
        self,
        token_embeddings: torch.Tensor,
        span_annotations: list[tuple[int, int]],
        document: str,
        max_length: Optional[int] = None,
        pooling: str = "mean",
    ) -> list[np.ndarray]:
        """Perform late chunking on token embeddings.

        Args:
            token_embeddings: Token-level embeddings [1, seq_len, hidden_size]
            span_annotations: Character-level span positions
            document: Original document for character-to-token mapping
            max_length: Maximum token length to process

        Returns:
            List of chunk embeddings as numpy arrays
        """
        if self._use_flag_model:
            logger.warning(
                "Late chunking with BGE-M3 FlagModel uses sentence-level fallback"
            )
            # Return sentence-level embeddings for each chunk
            embeddings = token_embeddings.cpu().numpy()
            return [embeddings[0, 0] for _ in span_annotations]

        # True late chunking with token-level embeddings
        # Note: This is a simplified mapping from character spans to token spans
        # A production implementation would need precise character-to-token alignment

        pooled_embeddings = []
        seq_len = token_embeddings.shape[1]
        doc_length = len(document)
        for char_start, char_end in span_annotations:
            token_start = int((char_start / doc_length) * seq_len)
            token_end = int((char_end / doc_length) * seq_len)
            token_start = max(0, token_start)
            token_end = min(seq_len, max(token_start + 1, token_end))
            if max_length is not None:
                token_end = min(token_end, max_length - 1)
            if token_end > token_start:
                chunk_tokens = token_embeddings[0, token_start:token_end]
                if pooling == "mean":
                    chunk_embedding = chunk_tokens.mean(dim=0)
                elif pooling == "max":
                    chunk_embedding = chunk_tokens.max(dim=0)[0]
                else:
                    raise ValueError(f"Unsupported pooling strategy: {pooling}")
                pooled_embeddings.append(chunk_embedding.detach().cpu().numpy())
        return pooled_embeddings

    def process_document(
        self, document: str, max_chunk_length: int = 500, pooling: str = "mean"
    ) -> tuple[list[str], list[np.ndarray]]:
        """Process a document with Late Chunking.

        Args:
            document: Input document text
            max_chunk_length: Maximum characters per chunk
            pooling: Pooling strategy ("mean" or "max")

        Returns:
            Tuple of (chunks, chunk_embeddings)
        """
        chunks, span_annotations = self.simple_sentence_chunker(
            document, max_chunk_length
        )
        token_embeddings = self.document_to_token_embeddings(document)
        chunk_embeddings = self.late_chunking(
            token_embeddings, span_annotations, document, pooling=pooling
        )
        logger.info(
            f"Processed document into {len(chunks)} chunks using Late Chunking (pooling={pooling})"
        )
        return chunks, chunk_embeddings

    def compare_with_traditional(
        self, document: str, chunks: list[str]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compare Late Chunking with traditional chunk-first embedding.

        Args:
            document: Full document text
            chunks: List of chunk texts

        Returns:
            Tuple of (late_chunking_embeddings, traditional_embeddings)
        """
        self.load_model()

        # Late chunking embeddings
        _, late_embeddings = self.process_document(document)

        # Traditional embeddings (embed each chunk separately)
        traditional_embeddings = []

        if self._use_flag_model:
            # Use BGE-M3 for individual chunks
            chunk_embeds = self.model.encode(
                chunks,
                batch_size=8,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"]
            traditional_embeddings = list(chunk_embeds)
        else:
            # Use transformers model for individual chunks
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    output = self.model(**inputs)
                    # Mean pooling
                    embedding = output.last_hidden_state.mean(dim=1)
                    traditional_embeddings.append(embedding.cpu().numpy()[0])

        return late_embeddings, traditional_embeddings
