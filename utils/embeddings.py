"""Embedding generation utilities using sentence-transformers."""

import os
from typing import Union, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers."""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: Optional[str] = None):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the model to use.
                       Defaults to environment variable or 'all-MiniLM-L6-v2'.
        """
        if self._model is None:
            self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            # Initialize sentence-transformers model
            self._model = SentenceTransformer(self.model_name)
            # Get dimension by embedding a dummy text
            dummy_emb = self._model.encode("test")
            self.embedding_dim = len(dummy_emb)
    
    def generate(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s).
        
        Args:
            text: Single text string or list of text strings.
            
        Returns:
            numpy array of embeddings.
        """
        if isinstance(text, str):
            embedding = self._model.encode(text)
            return np.array([embedding])
            
        embeddings = self._model.encode(text)
        return np.array(embeddings)
    
    def generate_single(self, text: str) -> list[float]:
        """Generate embedding for a single text and return as list.
        
        Args:
            text: Text string to embed.
            
        Returns:
            List of floats representing the embedding.
        """
        return self._model.encode(text).tolist()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score between 0 and 1.
        """
        # Embeddings from HuggingFaceEmbeddings are usually normalized
        return float(np.dot(embedding1, embedding2))
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
