"""PDF parsing and text extraction tools."""

import os
import re
import uuid
from io import BytesIO
from typing import BinaryIO, Optional, List, Tuple

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from models.schemas import DocumentChunk

load_dotenv()


class PDFParser:
    """Parse PDF documents and extract text chunks."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """Initialize the PDF parser.
        
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "50"))
    
    def extract_text(self, pdf_file: BinaryIO) -> list[tuple[int, str]]:
        """Extract text from PDF file.
        
        Args:
            pdf_file: PDF file object (file-like or BytesIO).
            
        Returns:
            List of (page_number, text) tuples.
        """
        reader = PdfReader(pdf_file)
        pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean up text
            text = self._clean_text(text)
            if text.strip():
                pages.append((i + 1, text))
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove weird characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Clean up extra spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def _split_text(self, text: str, page_number: int, document_id: str) -> list[DocumentChunk]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split.
            page_number: Source page number.
            document_id: Parent document ID.
            
        Returns:
            List of DocumentChunk objects.
        """
        if len(text) <= self.chunk_size:
            return [DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                content=text,
                page_number=page_number,
                chunk_index=0,
                metadata={
                    "source": "pdf",
                    "page": page_number
                }
            )]
        
        chunks = []
        separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Try to find a good split point
            end = start + self.chunk_size
            if end >= len(text):
                end = len(text)
            else:
                # Look for a separator near the end
                best_split = end
                for sep in separators:
                    if sep:
                        pos = text.rfind(sep, start, end)
                        if pos > start:
                            best_split = pos + len(sep)
                            break
                end = best_split
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = DocumentChunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "source": "pdf",
                        "page": page_number
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start forward with overlap
            start = end - self.chunk_overlap if end < len(text) else end
            
        return chunks
    
    def parse_pdf(
        self,
        pdf_file: BinaryIO,
        document_id: str
    ) -> tuple[list[DocumentChunk], int]:
        """Parse PDF and return chunks.
        
        Args:
            pdf_file: PDF file object.
            document_id: Unique document identifier.
            
        Returns:
            Tuple of (chunks list, page count).
        """
        pages = self.extract_text(pdf_file)
        all_chunks = []
        
        for page_number, text in pages:
            chunks = self._split_text(text, page_number, document_id)
            all_chunks.extend(chunks)
        
        # Update chunk indices to be global
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        
        return all_chunks, len(pages)
    
    def parse_pdf_bytes(
        self,
        pdf_bytes: bytes,
        document_id: str
    ) -> tuple[list[DocumentChunk], int]:
        """Parse PDF from bytes.
        
        Args:
            pdf_bytes: PDF file as bytes.
            document_id: Unique document identifier.
            
        Returns:
            Tuple of (chunks list, page count).
        """
        return self.parse_pdf(BytesIO(pdf_bytes), document_id)


def get_page_count(pdf_file: BinaryIO) -> int:
    """Get the page count of a PDF file.
    
    Args:
        pdf_file: PDF file object.
        
    Returns:
        Number of pages.
    """
    reader = PdfReader(pdf_file)
    return len(reader.pages)
