"""Vector store and document storage management."""

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import uuid

import numpy as np
import faiss
import pickle

from models.schemas import DocumentChunk, DocumentMetadata
from .embeddings import EmbeddingGenerator


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, dimension: Optional[int] = None, index_path: Optional[str] = None):
        """Initialize the vector store.
        
        Args:
            dimension: Embedding dimension (inferred from embedder if None).
            index_path: Path to save/load the FAISS index.
        """
        self.embedder = EmbeddingGenerator()
        self.dimension = dimension or self.embedder.dimension
        self.index_path = index_path or "data/vector_store"
        self.index = None
        self.chunks: Dict[str, DocumentChunk] = {}
        
        self._load_index()
    
    def _load_index(self):
        """Load existing index if available."""
        index_file = os.path.join(self.index_path, "index.faiss")
        chunks_file = os.path.join(self.index_path, "chunks.pkl")
        
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(chunks_file, "rb") as f:
                    self.chunks = pickle.load(f)
            except Exception as e:
                print(f"Failed to load index: {e}")
                self.index = None
                self.chunks = {}
    
    def save_index(self):
        """Save the FAISS index to disk."""
        if self.index is None:
            return
        
        os.makedirs(self.index_path, exist_ok=True)
        index_file = os.path.join(self.index_path, "index.faiss")
        chunks_file = os.path.join(self.index_path, "chunks.pkl")
        
        faiss.write_index(self.index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects.
            
        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.generate(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Initialize index if needed
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks metadata
        start_id = len(self.chunks)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{start_id + i}"
            self.chunks[chunk_id] = chunk
        
        self.save_index()
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_filter: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks.
        
        Args:
            query: Query text.
            top_k: Number of results to return.
            document_filter: Optional list of document IDs to filter by.
            
        Returns:
            List of (chunk, similarity_score) tuples.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_emb = self.embedder.generate(query)
        faiss.normalize_L2(query_emb)
        
        k = top_k * 2 if document_filter else top_k
        k = min(k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query_emb.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            
            chunk_id = str(idx)
            chunk = self.chunks.get(chunk_id)
            
            if chunk is None:
                continue
            
            # Apply document filter
            if document_filter and chunk.document_id not in document_filter:
                continue
            
            # Convert score to similarity (0-1 range)
            similarity = float(score)
            results.append((chunk, similarity))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.
        
        Args:
            document_id: ID of document to delete.
            
        Returns:
            Number of chunks deleted.
        """
        if self.index is None:
            return 0
        
        # Find chunks to delete
        chunks_to_delete = []
        chunks_to_keep = []
        
        for chunk_id, chunk in self.chunks.items():
            if chunk.document_id == document_id:
                chunks_to_delete.append(chunk_id)
            else:
                chunks_to_keep.append(chunk_id)
        
        if not chunks_to_delete:
            return 0
        
        # Rebuild index with remaining chunks
        if not chunks_to_keep:
            self.index = None
            self.chunks = {}
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
        else:
            # Rebuild index
            keep_chunks = [self.chunks[cid] for cid in chunks_to_keep]
            texts = [chunk.content for chunk in keep_chunks]
            embeddings = self.embedder.generate(texts)
            faiss.normalize_L2(embeddings)
            
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Rebuild chunks dict with new indices
            self.chunks = {}
            for i, chunk in enumerate(keep_chunks):
                self.chunks[str(i)] = chunk
            
            self.save_index()
        
        return len(chunks_to_delete)
    
    @property
    def total_chunks(self) -> int:
        """Get total number of indexed chunks."""
        if self.index:
            return self.index.ntotal
        return 0


class DocumentStore:
    """Store for managing uploaded documents and their metadata."""
    
    def __init__(self, storage_dir: str = "data/uploaded_docs"):
        """Initialize the document store.
        
        Args:
            storage_dir: Directory for storing uploaded documents.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "documents.json"
        
        self.documents: Dict[str, DocumentMetadata] = {}
        self._load_documents()
    
    def _load_documents(self):
        """Load document metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                for doc_id, doc_data in data.items():
                    doc_data["upload_time"] = datetime.fromisoformat(doc_data["upload_time"])
                    self.documents[doc_id] = DocumentMetadata(**doc_data)
    
    def _save_documents(self):
        """Save document metadata to disk."""
        data = {}
        for doc_id, doc in self.documents.items():
            doc_dict = doc.model_dump()
            doc_dict["upload_time"] = doc_dict["upload_time"].isoformat()
            data[doc_id] = doc_dict
        
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_document(
        self,
        filename: str,
        content: bytes,
        page_count: int = 0,
        chunk_count: int = 0
    ) -> DocumentMetadata:
        """Add a new document to the store.
        
        Args:
            filename: Original filename.
            content: File content as bytes.
            page_count: Number of pages in document.
            chunk_count: Number of indexed chunks.
            
        Returns:
            DocumentMetadata for the stored document.
        """
        doc_id = str(uuid.uuid4())
        file_path = self.storage_dir / f"{doc_id}_{filename}"
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create metadata
        metadata = DocumentMetadata(
            filename=filename,
            file_path=str(file_path),
            page_count=page_count,
            chunk_count=chunk_count,
            file_size_bytes=len(content),
        )
        
        self.documents[doc_id] = metadata
        self._save_documents()
        
        return metadata
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[Tuple[str, DocumentMetadata]]:
        """List all documents.
        
        Returns:
            List of (doc_id, metadata) tuples.
        """
        return list(self.documents.items())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document.
        
        Args:
            doc_id: Document ID to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        if doc_id not in self.documents:
            return False
        
        # Delete file
        doc = self.documents[doc_id]
        try:
            os.remove(doc.file_path)
        except FileNotFoundError:
            pass
        
        # Remove from metadata
        del self.documents[doc_id]
        self._save_documents()
        
        return True
    
    def update_chunk_count(self, doc_id: str, chunk_count: int):
        """Update the chunk count for a document."""
        if doc_id in self.documents:
            self.documents[doc_id].chunk_count = chunk_count
            self._save_documents()
