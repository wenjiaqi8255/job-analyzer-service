import logging
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from cachetools import LRUCache

logger = logging.getLogger(__name__)

class BatchEmbeddingProcessor:
    """
    Handles batch encoding of text chunks with caching to avoid re-computation.
    """
    def __init__(self, embedding_model: SentenceTransformer, batch_size: int = 32, cache_size: int = 1000):
        if not isinstance(embedding_model, SentenceTransformer):
            raise TypeError("embedding_model must be an instance of SentenceTransformer")
        
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.cache = LRUCache(maxsize=cache_size)

    def encode_chunks(self, chunks: List[str]) -> torch.Tensor:
        """
        Encodes a list of text chunks, using a cache to retrieve existing embeddings
        and batching the rest.
        
        Args:
            chunks: A list of strings to be encoded.
            
        Returns:
            A torch.Tensor containing the embeddings for all chunks.
        """
        if not chunks:
            return torch.empty(0)

        # Separate cached and non-cached chunks
        cached_embeddings = {}
        chunks_to_encode = []
        
        for i, chunk in enumerate(chunks):
            if chunk in self.cache:
                cached_embeddings[i] = self.cache[chunk]
            else:
                chunks_to_encode.append((i, chunk))

        logger.info(f"Embedding {len(chunks)} chunks: {len(cached_embeddings)} from cache, {len(chunks_to_encode)} to encode.")

        # Batch encode non-cached chunks
        if chunks_to_encode:
            # Extract the text from the list of (index, text) tuples
            texts_to_encode = [text for _, text in chunks_to_encode]
            try:
                new_embeddings = self.embedding_model.encode(
                    texts_to_encode,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False 
                )
                
                # Add new embeddings to cache and to the results
                for (original_index, chunk_text), embedding in zip(chunks_to_encode, new_embeddings):
                    self.cache[chunk_text] = embedding
                    cached_embeddings[original_index] = embedding

            except Exception as e:
                logger.error(f"Failed to encode batch: {e}", exc_info=True)
                # Handle failure: maybe return empty or partial results
                # For now, we continue with what we have
                pass

        # Reconstruct the final tensor in the original order
        final_embeddings = [cached_embeddings[i] for i in range(len(chunks))]
        
        # Stack the list of tensors into a single tensor
        return torch.stack(final_embeddings) 