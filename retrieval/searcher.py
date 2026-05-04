"""
searcher.py — Search engine untuk RAG: query ke Qdrant, ambil chunks relevan

Apa itu Retrieval?
  Retrieval adalah proses mencari informasi yang relevan dari database berdasarkan
  pertanyaan user. Di RAG (Retrieval-Augmented Generation), ini adalah langkah
  pertama sebelum LLM menjawab.
  
  Alur RAG:
  1. User tanya: "Apa risiko klausul indemnifikasi?"
  2. RETRIEVAL: Cari 5 pasal paling relevan dari Qdrant
  3. GENERATION: Kasih pasal-pasal tersebut ke LLM untuk dijawab

Cosine Similarity dalam Retrieval:
  Query "risiko klausul indemnifikasi" → embed jadi vector
  Cari 5 chunks dengan vector paling "dekat" (similarity tinggi)
  → Qdrant otomatis urutkan dari similarity tertinggi ke terendah
"""

import logging
from typing import Optional

import numpy as np

from ingestion.embedder import LegalEmbedder, get_default_embedder
from ingestion.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── SearchResult ────────────────────────────────────────────────────────────

class SearchResult:
    """
    Hasil pencarian untuk satu chunk yang relevan.
    
    Attributes:
        chunk_id    : ID chunk (misal: kontrak_sewa_pasal_5)
        score       : Similarity score (0.0 - 1.0, makin tinggi makin relevan)
        text        : Isi teks pasal
        pasal_number: Nomor pasal
        pasal_title : Judul pasal
        source_file : Nama file asal
        metadata    : Dict tambahan (jika ada)
    """
    def __init__(self, result_dict: dict):
        self.chunk_id = result_dict.get("chunk_id", "")
        self.score = result_dict.get("score", 0.0)
        self.text = result_dict.get("text", "")
        self.pasal_number = result_dict.get("pasal_number", 0)
        self.pasal_title = result_dict.get("pasal_title", "")
        self.source_file = result_dict.get("source_file", "")
        self.metadata = result_dict.get("metadata", {})
        self.ayat_number = result_dict.get("ayat_number")
        self.level = result_dict.get("level")
    
    def __repr__(self) -> str:
        return (
            f"<SearchResult chunk_id={self.chunk_id} "
            f"score={self.score:.3f} pasal={self.pasal_number}>"
        )
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text,
            "pasal_number": self.pasal_number,
            "pasal_title": self.pasal_title,
            "ayat_number": self.ayat_number,   # ✅ tambahan
            "level": self.level,               # ✅ tambahan
            "source_file": self.source_file,
        }
    
    def format_for_llm(self) -> str:
        """
        Format chunk untuk diberikan ke LLM sebagai context.

        Format baru:
        ---
        [Pasal 5 Ayat 2 - Pembayaran]
        (Sumber: Kontrak_Sewa.pdf | Relevance: 0.87)

        Isi ayat di sini...
        ---
        """

        # 🔥 Tambahkan info ayat kalau ada
        ayat_info = f" Ayat {self.ayat_number}" if getattr(self, "ayat_number", None) else ""

        return (
            f"---\n"
            f"[Pasal {self.pasal_number}{ayat_info} - {self.pasal_title}]\n"
            f"(Sumber: {self.source_file} | Relevance: {self.score:.2f})\n\n"
            f"{self.text}\n"
            f"---"
        )


# ─── Searcher Class ──────────────────────────────────────────────────────────

class LegalDocSearcher:
    """
    Search engine untuk mencari pasal-pasal relevan dari Qdrant.
    
    Cara kerja:
    1. Terima query dari user (string)
    2. Embed query menjadi vector menggunakan embedder
    3. Kirim vector ke Qdrant untuk search
    4. Return top-k chunks paling relevan
    
    Usage:
        searcher = LegalDocSearcher()
        results = searcher.search("Apa syarat pembayaran?", top_k=3)
        for r in results:
            print(r.pasal_title, "→", r.score)
    """
    
    def __init__(
        self,
        embedder: Optional[LegalEmbedder] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ):
        """
        Inisialisasi searcher dengan embedder dan vector store.
        
        Args:
            embedder    : LegalEmbedder instance. Jika None, akan dibuat baru.
            vector_store: QdrantVectorStore instance. Jika None, akan dibuat baru.
        """
        # Load embedder (pakai yang sudah ada atau buat baru)
        if embedder is None:
            logger.info("Inisialisasi embedder untuk search...")
            self.embedder = get_default_embedder()
        else:
            self.embedder = embedder
        
        # Load vector store (konek ke Qdrant)
        if vector_store is None:
            logger.info("Koneksi ke Qdrant untuk search...")
            self.vector_store = QdrantVectorStore()
        else:
            self.vector_store = vector_store
        
        logger.info("✓ Searcher siap digunakan!")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filter_source: Optional[str] = None,
        filter_pasal: Optional[int] = None,
        filter_ayat: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Cari chunks paling relevan berdasarkan query.
        
        Args:
            query          : Pertanyaan atau teks yang ingin dicari
            top_k          : Jumlah hasil yang dikembalikan (default: 5)
            score_threshold: Minimum similarity score (0.0 - 1.0, default: 0.5)
                             Chunks dengan score < threshold akan dibuang
            filter_source  : Filter berdasarkan nama file tertentu (opsional)
                             Contoh: "Kontrak_Sewa.pdf"
        
        Returns:
            List of SearchResult, diurutkan dari score tertinggi ke terendah
            
        Example:
            results = searcher.search(
                query="Bagaimana cara terminate kontrak lebih awal?",
                top_k=3,
                score_threshold=0.6,
            )
        """
        logger.info(f"Search query: '{query[:60]}...' (top_k={top_k}, threshold={score_threshold})")
        
        # Step 1: Embed query menjadi vector
        query_vector = self.embedder.embed_text(query)
        
        # Step 2: Search di Qdrant
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_source=filter_source,
            filter_pasal=filter_pasal,
            filter_ayat=filter_ayat,
        )
        
        # Step 3: Convert ke SearchResult objects
        results = [SearchResult(r) for r in raw_results]

        # 🔥 BOOST PREAMBLE
        for r in results:
            if r.pasal_number == 0:
                r.score += 0.15  # bisa adjust (0.1 - 0.3)
        
        logger.info(f"  → Ditemukan {len(results)} chunks relevan")
        for i, r in enumerate(results[:3], 1):  # Log top 3
            logger.info(f"    {i}. [{r.source_file}] Pasal {r.pasal_number} (score={r.score:.3f})")
        
        return results
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 4000,
    ) -> str:
        """
        Ambil context untuk diberikan ke LLM dalam format yang siap pakai.
        
        Fungsi ini menggabungkan:
        1. Search chunks relevan
        2. Format setiap chunk dengan metadata
        3. Gabungkan jadi satu string dengan delimiter jelas
        4. Truncate jika terlalu panjang (agar tidak melebihi context window LLM)
        
        Args:
            query             : Pertanyaan user
            top_k             : Jumlah chunks yang diambil
            max_context_length: Maksimum panjang karakter context
            
        Returns:
            String context siap untuk prompt LLM
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "Tidak ada pasal yang relevan ditemukan dalam kontrak."
        
        # Format setiap chunk
        formatted_chunks = [r.format_for_llm() for r in results]
        
        # Gabungkan dengan double newline
        context = "\n\n".join(formatted_chunks)
        
        # Truncate jika terlalu panjang
        if len(context) > max_context_length:
            logger.warning(
                f"Context terlalu panjang ({len(context)} chars), "
                f"di-truncate ke {max_context_length} chars"
            )
            context = context[:max_context_length] + "\n\n[... truncated ...]"
        
        return context


def get_default_searcher() -> LegalDocSearcher:
    """Shortcut untuk mendapatkan searcher dengan konfigurasi default."""
    return LegalDocSearcher()


if __name__ == "__main__":
    """Test searcher - pastikan sudah jalankan ingestion pipeline dulu"""
    import sys
    
    print("="*60)
    print("TEST SEARCHER")
    print("="*60)
    
    print("\n[1] Inisialisasi searcher...")
    try:
        searcher = LegalDocSearcher()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  → Pastikan Qdrant sudah jalan: docker-compose up -d qdrant")
        print("  → Pastikan sudah jalankan ingestion: python -m ingestion.vector_store")
        sys.exit(1)
    
    print("\n[2] Test search...")
    test_queries = [
        "Bagaimana cara pembayaran dalam kontrak ini?",
        "Apa yang terjadi jika terlambat bayar?",
        "Siapa yang bertanggung jawab atas kerusakan?",
    ]
    
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = searcher.search(q, top_k=2, score_threshold=0.3)
        
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. Pasal {r.pasal_number} - {r.pasal_title} (score={r.score:.3f})")
                print(f"     File: {r.source_file}")
                print(f"     Preview: {r.text[:100]}...")
        else:
            print("  → Tidak ada hasil ditemukan")
    
    print("\n[3] Test get_context_for_llm...")
    context = searcher.get_context_for_llm("Apa hak dan kewajiban penyewa?", top_k=3)
    print(f"Context length: {len(context)} characters")
    print(f"Preview:\n{context[:300]}...\n")
    
    print("="*60)
    print("✓ Test selesai!")