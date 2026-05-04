"""
ingestion/__init__.py

Package ingestion — berisi semua komponen untuk memproses dokumen
dan menyimpannya ke vector store.

Modul yang tersedia:
- ocr.py        : Ekstrak teks dari PDF/DOCX (dengan OCR fallback)
- chunker.py    : Pecah teks menjadi chunks per pasal
- embedder.py   : Ubah teks menjadi vektor embedding (Legal-BERT)
- vector_store.py: Simpan & cari vektor di Qdrant

Cara pakai cepat (one-liner pipeline):
    from ingestion.vector_store import IngestionPipeline, get_vector_store
    from ingestion.embedder import get_embedder

    pipeline = IngestionPipeline(
        vector_store=get_vector_store(),
        embedder=get_embedder(),
    )
    result = pipeline.ingest_file("data/raw/kontrak.pdf")
"""

from ingestion.chunker import chunk_pdf, extract_text_from_pdf
from ingestion.embedder import LegalEmbedder
from ingestion.vector_store import QdrantVectorStore

__all__ = [
    # OCR
    "extract_text",
    "extract_text_from_pdf",
    "extract_text_from_docx",
    # Chunker
    "chunk_document",
    "DocumentChunk",
    "get_chunks_summary",
    # Embedder
    "LegalEmbedder",
    "get_embedder",
    "create_embedder",
    # Vector Store
    "QdrantVectorStore",
    "IngestionPipeline",
    "SearchResult",
    "get_vector_store",
    "create_vector_store",
]