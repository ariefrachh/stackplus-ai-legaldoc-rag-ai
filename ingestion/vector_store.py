"""
vector_store.py — Setup Qdrant lokal dan simpan chunks dengan embedding-nya

Apa itu Qdrant?
  Qdrant adalah "Vector Database" — database yang didesain khusus untuk menyimpan
  dan mencari vector (embedding). Tidak seperti database biasa yang cari berdasarkan
  kata kunci, Qdrant bisa cari berdasarkan "kemiripan makna" (semantic search).

  Contoh:
  Query: "denda keterlambatan" → Qdrant cari pasal yang punya makna paling dekat,
  bahkan jika pasal tersebut tidak mengandung kata "denda" atau "keterlambatan" secara persis.

Cara kerja file ini:
  1. Konek ke Qdrant (via Docker di localhost:6333)
  2. Buat "collection" (seperti tabel di database biasa)
  3. Simpan (upsert) embedding + metadata ke collection
  4. Menyediakan fungsi search untuk retrieval nanti
"""

import logging
from typing import Optional
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Konstanta ──────────────────────────────────────────────────────────────────

COLLECTION_NAME = settings.qdrant_collection
VECTOR_DIM_LEGAL_BERT = settings.embedding_dimension
DEFAULT_QDRANT_HOST = settings.qdrant_host
DEFAULT_QDRANT_PORT = settings.qdrant_port


# ─── VectorStore Class ──────────────────────────────────────────────────────────

class QdrantVectorStore:
    """
    Wrapper di atas Qdrant client untuk operasi simpan dan cari chunks.
    
    Kenapa dibungkus class?
    Supaya jika di masa depan kita ganti ke database lain (Pinecone, Weaviate, dll.),
    kita hanya perlu ganti class ini, tidak perlu ubah kode di tempat lain.
    
    Usage:
        store = QdrantVectorStore()
        store.setup_collection(vector_dim=768)
        store.upsert(embedded_chunks)
        results = store.search(query_vector, top_k=5)
    """
    
    def __init__(
        self,
        host: str = DEFAULT_QDRANT_HOST,
        port: int = DEFAULT_QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
    ):
        """
        Inisialisasi koneksi ke Qdrant.
        
        Args:
            host           : Host Qdrant (default: localhost)
            port           : Port Qdrant (default: 6333)
            collection_name: Nama collection yang akan dipakai
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        
        self._connect()
    
    def _connect(self) -> None:
        """Buat koneksi ke Qdrant server."""
        logger.info(f"Menghubungkan ke Qdrant di {self.host}:{self.port}...")
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Test koneksi dengan list collections
            self.client.get_collections()
            logger.info("  ✓ Berhasil terhubung ke Qdrant!")
        except Exception as e:
            raise ConnectionError(
                f"Gagal terhubung ke Qdrant di {self.host}:{self.port}.\n"
                f"Pastikan Docker Qdrant sudah jalan: docker-compose up -d qdrant\n"
                f"Detail error: {e}"
            ) from e
    
    def collection_exists(self) -> bool:
        """Cek apakah collection sudah ada."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
    
    def setup_collection(
        self,
        vector_dim: int = VECTOR_DIM_LEGAL_BERT,
        recreate: bool = False,
    ) -> None:
        """
        Buat collection di Qdrant jika belum ada.
        
        Apa itu collection?
        Analoginya seperti "tabel" di SQL. Di dalamnya kita simpan:
        - vector (embedding)
        - payload (metadata: pasal_number, pasal_title, text, dll.)
        
        Args:
            vector_dim: Dimensi vector embedding (768 untuk legal-BERT)
            recreate  : Jika True, hapus collection lama dan buat baru.
                        HATI-HATI: ini akan menghapus semua data!
        
        Raises:
            RuntimeError jika client belum terkoneksi
        """
        if self.client is None:
            raise RuntimeError("Client belum terkoneksi.")
        
        if self.collection_exists():
            if recreate:
                logger.warning(f"  ⚠️  Menghapus collection '{self.collection_name}' yang sudah ada...")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"  → Collection '{self.collection_name}' sudah ada, skip pembuatan.")
                return
        
        logger.info(f"Membuat collection '{self.collection_name}' (dim={vector_dim})...")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_dim,
                # Cosine distance: ukur kemiripan berdasarkan sudut antara dua vector
                # Cocok untuk semantic similarity (hasil embedding yang sudah dinormalisasi)
                distance=qmodels.Distance.COSINE,
            ),
        )
        
        logger.info(f"  ✓ Collection '{self.collection_name}' berhasil dibuat!")
    
    def upsert(self, embedded_chunks: list[dict], batch_size: int = 64) -> int:
        """
        Simpan (upsert) embedded chunks ke Qdrant.
        
        "Upsert" = Update + Insert:
        - Jika chunk_id sudah ada → update
        - Jika belum ada → insert baru
        Ini memastikan tidak ada duplikat jika kita jalankan pipeline dua kali.
        
        Args:
            embedded_chunks: List of dict dari embedder.embed_chunks()
                             Format: [{"id": str, "vector": ndarray, "payload": dict}, ...]
            batch_size     : Jumlah chunk per batch upsert (untuk efisiensi)
            
        Returns:
            Jumlah chunk yang berhasil di-upsert
        """
        if not embedded_chunks:
            logger.warning("Tidak ada chunk yang di-upsert (list kosong).")
            return 0
        
        total = len(embedded_chunks)
        logger.info(f"Menyimpan {total} chunks ke Qdrant (batch_size={batch_size})...")
        
        # Proses dalam batch agar tidak overload memory
        success_count = 0
        for i in range(0, total, batch_size):
            batch = embedded_chunks[i : i + batch_size]
            
            # Konversi ke format Qdrant PointStruct
            points = []
            for item in batch:
                vector = item["vector"]
                # Pastikan vector adalah list of float (bukan numpy array)
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                
                points.append(
                    qmodels.PointStruct(
                        id=self._make_numeric_id(item["id"]),  # Qdrant perlu int atau UUID
                        vector=vector,
                        payload=item["payload"],  # Semua metadata disimpan di sini
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,  # Tunggu sampai upsert selesai sebelum lanjut
            )
            success_count += len(batch)
            logger.info(f"  → Batch {i//batch_size + 1}: {len(batch)} points di-upsert")
        
        logger.info(f"  ✓ Total {success_count} chunks berhasil disimpan ke Qdrant!")
        return success_count
    
    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_source: Optional[str] = None,
        filter_pasal: Optional[int] = None,
        filter_ayat: Optional[int] = None,
    ) -> list[dict]:
        """
        Semantic + metadata-aware search di Qdrant.

        Support:
        - Semantic similarity
        - Filter by source_file
        - Filter by pasal_number
        - Filter by ayat_number
        """

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # 🔥 Build dynamic filter
        must_conditions = []

        if filter_source:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="source",
                    match=qmodels.MatchValue(value=filter_source),
                )
            )

        if filter_pasal is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="pasal_number",
                    match=qmodels.MatchValue(value=filter_pasal),
                )
            )

        if filter_ayat is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="ayat_number",
                    match=qmodels.MatchValue(value=filter_ayat),
                )
            )

        search_filter = (
            qmodels.Filter(must=must_conditions)
            if must_conditions else None
        )

        # 🔍 Search ke Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=True,
        )

        # 📦 Format hasil
        results = []
        for hit in search_results:
            payload = hit.payload or {}

            results.append({
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "score": round(hit.score, 4),
                "text": payload.get("text", ""),
                "pasal_number": payload.get("pasal_number", 0),
                "pasal_title": payload.get("pasal_title", ""),
                "ayat_number": payload.get("ayat_number"),
                "level": payload.get("level"),
                "source_file": payload.get("source", ""),
            })

        return results
    
    def get_collection_info(self) -> dict:
        """Ambil informasi tentang collection (jumlah vectors, status, dll.)"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
            "vector_size": info.config.params.vectors.size,
        }
    
    def _make_numeric_id(self, chunk_id: str):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# ─── Fungsi Pipeline Lengkap ────────────────────────────────────────────────────

def run_ingestion_pipeline(
    pdf_directory: str,
    qdrant_host: str = DEFAULT_QDRANT_HOST,
    qdrant_port: int = DEFAULT_QDRANT_PORT,
    recreate_collection: bool = False,
    model_name: Optional[str] = None,
) -> dict:
    """
    Pipeline lengkap: PDF → Chunk → Embed → Simpan ke Qdrant.
    
    Ini adalah fungsi "one-stop-shop" yang memanggil chunker, embedder, 
    dan vector_store secara berurutan.
    
    Args:
        pdf_directory      : Path folder yang berisi file PDF
        qdrant_host        : Host Qdrant
        qdrant_port        : Port Qdrant
        recreate_collection: Hapus dan buat ulang collection
        model_name         : Override nama model embedding (None = pakai default)
        
    Returns:
        Dict summary hasil ingestion:
        {
            "total_chunks": int,
            "total_upserted": int,
            "collection_info": dict,
            "source_files": list[str],
        }
    """
    from ingestion.chunker import chunk_directory
    from ingestion.embedder import LegalEmbedder, DEFAULT_MODEL_NAME
    
    logger.info("="*60)
    logger.info("MULAI INGESTION PIPELINE")
    logger.info("="*60)
    
    # Step 1: Chunking
    logger.info("\n[Step 1/3] Chunking PDF...")
    chunks = chunk_directory(pdf_directory)
    if not chunks:
        raise ValueError(f"Tidak ada chunk yang dihasilkan dari direktori: {pdf_directory}")
    
    # Step 2: Embedding
    logger.info("\n[Step 2/3] Embedding chunks...")
    embedder = LegalEmbedder(
        model_name="nlpaueb/legal-bert-base-uncased",
        use_gpu=False
    )
    embedded = embedder.embed_chunks(chunks)
    vector_dim = embedder.embedding_dim
    
    # Step 3: Simpan ke Qdrant
    logger.info("\n[Step 3/3] Menyimpan ke Qdrant...")
    store = QdrantVectorStore(host=qdrant_host, port=qdrant_port)
    store.setup_collection(vector_dim=vector_dim, recreate=recreate_collection)
    upserted = store.upsert(embedded)
    
    # Summary
    info = store.get_collection_info()
    source_files = list({c.source_file for c in chunks})
    
    summary = {
        "total_chunks": len(chunks),
        "total_upserted": upserted,
        "collection_info": info,
        "source_files": source_files,
    }
    
    logger.info("\n" + "="*60)
    logger.info("✓ INGESTION PIPELINE SELESAI!")
    logger.info(f"  Chunks    : {summary['total_chunks']}")
    logger.info(f"  Upserted  : {summary['total_upserted']}")
    logger.info(f"  Files     : {', '.join(source_files)}")
    logger.info("="*60)
    
    return summary


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Jalankan dengan: python -m ingestion.vector_store
    Akan test koneksi ke Qdrant dan jalankan full pipeline.
    
    Pastikan Qdrant Docker sudah jalan dulu:
    docker-compose up -d qdrant
    """
    import sys
    from pathlib import Path
    
    # Test koneksi dulu
    print("\n[TEST] Cek koneksi Qdrant...")
    try:
        store = QdrantVectorStore()
        print("  ✓ Qdrant terhubung!")
    except ConnectionError as e:
        print(f"  ✗ {e}")
        sys.exit(1)
    
    # Test full pipeline
    data_dir = Path("data/dummy")
    if not data_dir.exists():
        print(f"  ✗ Folder {data_dir} tidak ditemukan. Jalankan dari root project.")
        sys.exit(1)
    
    summary = run_ingestion_pipeline(
        pdf_directory=str(data_dir),
        recreate_collection=True,
    )
    
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k}: {v}")