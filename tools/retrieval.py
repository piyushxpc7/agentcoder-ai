"""Vector search and retrieval tools."""

from typing import Optional, List, Dict

from models.schemas import DocumentChunk
from utils.storage import VectorStore, DocumentStore


class RetrievalTools:
    """Tools for document retrieval and search."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        document_store: Optional[DocumentStore] = None
    ):
        """Initialize retrieval tools.
        
        Args:
            vector_store: VectorStore instance.
            document_store: DocumentStore instance.
        """
        self.vector_store = vector_store or VectorStore()
        self.document_store = document_store or DocumentStore()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[Dict]:
        """Search for relevant document chunks.
        
        Args:
            query: Search query.
            top_k: Maximum number of results.
            document_ids: Optional filter by document IDs.
            min_score: Minimum similarity score threshold.
            
        Returns:
            List of result dictionaries with chunk info and scores.
        """
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            document_filter=document_ids
        )
        
        formatted_results = []
        for chunk, score in results:
            if score < min_score:
                continue
            
            # Get document metadata
            doc_meta = None
            for doc_id, meta in self.document_store.list_documents():
                if doc_id == chunk.document_id:
                    doc_meta = meta
                    break
            
            formatted_results.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_name": doc_meta.filename if doc_meta else "Unknown",
                "content": chunk.content,
                "page_number": chunk.page_number,
                "score": round(score, 4),
            })
        
        return formatted_results
    
    def get_context(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> str:
        """Get context string for RAG.
        
        Args:
            query: Search query.
            top_k: Number of chunks to retrieve.
            document_ids: Optional filter by document IDs.
            
        Returns:
            Formatted context string.
        """
        results = self.search(query, top_k, document_ids)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['document_name']}, Page {result['page_number']}]\n"
                f"{result['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_document_summary(self, document_id: str) -> Optional[Dict]:
        """Get summary information about a document.
        
        Args:
            document_id: Document ID.
            
        Returns:
            Document summary dictionary or None.
        """
        doc = self.document_store.get_document(document_id)
        if not doc:
            return None
        
        return {
            "filename": doc.filename,
            "page_count": doc.page_count,
            "chunk_count": doc.chunk_count,
            "file_size_kb": round(doc.file_size_bytes / 1024, 2),
            "upload_time": doc.upload_time.isoformat(),
        }
    
    def list_all_documents(self) -> List[Dict]:
        """List all indexed documents.
        
        Returns:
            List of document summary dictionaries.
        """
        documents = []
        for doc_id, meta in self.document_store.list_documents():
            documents.append({
                "id": doc_id,
                "filename": meta.filename,
                "page_count": meta.page_count,
                "chunk_count": meta.chunk_count,
                "file_size_kb": round(meta.file_size_bytes / 1024, 2),
                "upload_time": meta.upload_time.isoformat(),
            })
        return documents
