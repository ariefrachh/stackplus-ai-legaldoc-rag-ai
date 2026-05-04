"""
embedder.py — Load model legal-BERT dan generate embedding per chunk

Apa itu embedding?
  Embedding adalah representasi teks dalam bentuk angka (vector).
  Teks yang artinya mirip akan punya vector yang "berdekatan" di ruang vektor.
  
  Contoh:
  "klausul pembayaran" → [0.12, -0.45, 0.87, ...]  (768 angka)
  "syarat bayar"      → [0.11, -0.43, 0.85, ...]  (angka-angkanya mirip!)
  "kucing lucu"       → [0.91,  0.33, -0.12, ...] (sangat berbeda)

Kenapa pakai legal-BERT bukan BERT biasa?
  Model legal-BERT sudah "belajar" dari banyak teks hukum (kontrak, putusan, dll.)
  sehingga lebih paham istilah-istilah legal dibanding BERT generik.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.chunker import Chunk
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Konstanta ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = settings.embedding_model
EXPECTED_DIMENSION = settings.embedding_dimension
DEFAULT_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512


# ─── EmbedderModel Class ────────────────────────────────────────────────────────

class LegalEmbedder:
    """
    Wrapper untuk SentenceTransformer model legal-BERT.
    
    Kenapa pakai class bukan function biasa?
    Supaya model hanya di-load SEKALI ke memori, tidak setiap kali mau embed.
    Load model itu lambat (~10-30 detik), jadi kita load sekali, pakai berkali-kali.
    
    Usage:
        embedder = LegalEmbedder()
        vectors = embedder.embed_chunks(chunks)
    """
    
    def __init__(self, model_name=DEFAULT_MODEL_NAME, use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.embedding_dim = None

        self._load_model()  # ✅ tanpa argumen
    
    def _load_model(self) -> None:
        """Load model TANPA fallback. Fail fast kalau ada masalah."""
        
        device = "cuda" if self.use_gpu else "cpu"
        
        logger.info(f"Loading model: '{self.model_name}' (device={device})")
        logger.info("  → Pertama kali download bisa makan waktu beberapa menit...")
        
        start = time.time()
        
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            self.model.max_seq_length = MAX_SEQ_LENGTH

            # cek dimensi
            dummy = self.model.encode(["test"], show_progress_bar=False)
            self.embedding_dim = dummy.shape[1]

            if self.embedding_dim != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch! "
                    f"Expected {EXPECTED_DIMENSION}, got {self.embedding_dim}"
                )

            elapsed = time.time() - start
            logger.info(f"  ✓ Model loaded! Dimensi: {self.embedding_dim}, Waktu: {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"  ✗ Gagal load model '{self.model_name}': {e}")
            raise RuntimeError(
                "Model gagal diload. Tidak ada fallback.\n"
                "Cek:\n"
                "- koneksi internet\n"
                "- nama model benar\n"
                "- huggingface tidak timeout"
            ) from e
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed satu teks menjadi vector.
        
        Args:
            text: Teks yang ingin di-embed
            
        Returns:
            numpy array shape (embedding_dim,) — misal (768,) untuk legal-BERT
        """
        if self.model is None:
            raise RuntimeError("Model belum ter-load. Inisialisasi LegalEmbedder terlebih dahulu.")
        
        # Truncate teks jika terlalu panjang (akan otomatis di-handle model, tapi kita log saja)
        if len(text.split()) > MAX_SEQ_LENGTH:
            logger.debug(f"Teks terlalu panjang ({len(text.split())} kata), akan di-truncate ke {MAX_SEQ_LENGTH} token")
        
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalization agar cosine similarity = dot product
            show_progress_bar=False,
        )
        return vector
    
    def embed_batch(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """
        Embed banyak teks sekaligus (lebih efisien dari memanggil embed_text satu-satu).
        
        Args:
            texts     : List of string yang ingin di-embed
            batch_size: Jumlah teks per batch
            
        Returns:
            numpy array shape (len(texts), embedding_dim)
            Contoh: 10 chunk × 768 dim = array shape (10, 768)
        """
        if not texts:
            return np.array([])
        
        if self.model is None:
            raise RuntimeError("Model belum ter-load.")
        
        logger.info(f"Embedding {len(texts)} teks (batch_size={batch_size})...")
        start = time.time()
        
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,   # Tampilkan progress bar di terminal
        )
        
        elapsed = time.time() - start
        logger.info(f"  ✓ Selesai! Shape: {vectors.shape}, Waktu: {elapsed:.1f}s")
        return vectors
    
    def embed_chunks(self, chunks: list[Chunk], batch_size: int = DEFAULT_BATCH_SIZE) -> list[dict]:
        """
        Embed list of Chunk dan kembalikan list of dict siap simpan ke Qdrant.
        
        Proses:
        1. Ambil semua teks dari chunks
        2. Batch embed semuanya
        3. Pasangkan kembali setiap vector dengan metadata chunk-nya
        
        Args:
            chunks    : List of Chunk dari chunker.py
            batch_size: Jumlah chunk per batch untuk embedding
            
        Returns:
            List of dict, masing-masing berisi:
            - "id"      : chunk_id (string)
            - "vector"  : embedding vector (numpy array)
            - "payload" : metadata (dict, untuk disimpan di Qdrant)
            
        Example:
            embedder = LegalEmbedder()
            result = embedder.embed_chunks(chunks)
            # result[0] = {"id": "kontrak_sewa_pasal_1", "vector": array([...]), "payload": {...}}
        """
        if not chunks:
            logger.warning("List chunks kosong, tidak ada yang perlu di-embed.")
            return []
        
        logger.info(f"Mulai embed {len(chunks)} chunks...")
        
        # Ambil teks dari setiap chunk
        # Kita gabungkan judul + teks agar embedding mencerminkan konteks pasal
        texts = [
            f"{c.pasal_title}\n\n{c.text}"
            for c in chunks
        ]
        
        # Batch embed semua teks
        vectors = self.embed_batch(texts, batch_size=batch_size)
        
        # Gabungkan vector dengan metadata chunk
        results = []
        for chunk, vector in zip(chunks, vectors):
            results.append({
                "id": chunk.chunk_id,
                "vector": vector,
                "payload": chunk.to_dict(),
            })
        
        logger.info(f"  ✓ {len(results)} embedding siap disimpan ke Qdrant")
        return results


# ─── Fungsi Helper (tanpa harus inisialisasi class) ─────────────────────────────

def get_default_embedder() -> LegalEmbedder:
    """
    Shortcut untuk mendapatkan embedder dengan konfigurasi default.
    Cocok untuk dipakai di modul lain (searcher.py, dll.)
    
    Usage:
        from ingestion.embedder import get_default_embedder
        embedder = get_default_embedder()
        vector = embedder.embed_text("klausul pembayaran denda keterlambatan")
    """
    return LegalEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        use_gpu=False,
    )


def embed_query(query: str, embedder: Optional[LegalEmbedder] = None) -> np.ndarray:
    """
    Embed satu query string (untuk keperluan search di retrieval/searcher.py).
    Jika embedder tidak diberikan, akan buat baru (note: lambat karena harus load model).
    
    Args:
        query   : Pertanyaan atau teks yang ingin dicari
        embedder: LegalEmbedder instance yang sudah ada (opsional)
        
    Returns:
        numpy array shape (embedding_dim,)
    """
    if embedder is None:
        embedder = get_default_embedder()
    return embedder.embed_text(query)


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Jalankan dengan: python -m ingestion.embedder
    Akan test embed beberapa contoh teks hukum.
    """
    print("="*60)
    print("TEST EMBEDDER")
    print("="*60)
    
    # Test 1: Load model
    print("\n[1] Loading model legal-BERT...")
    embedder = LegalEmbedder()
    print(f"    Model: {embedder.model_name}")
    print(f"    Dimensi: {embedder.embedding_dim}")
    
    # Test 2: Embed single text
    print("\n[2] Embed single text...")
    sample_text = "Pihak pertama wajib membayar denda keterlambatan sebesar 2% per bulan."
    vector = embedder.embed_text(sample_text)
    print(f"    Input : '{sample_text[:50]}...'")
    print(f"    Output: array shape={vector.shape}, dtype={vector.dtype}")
    print(f"    Sample: [{vector[0]:.4f}, {vector[1]:.4f}, ...]")
    
    # Test 3: Cosine similarity antara dua teks mirip vs tidak mirip
    print("\n[3] Cosine Similarity Test...")
    text_a = "pembayaran denda keterlambatan"
    text_b = "biaya penalti keterlambatan bayar"   # mirip maknanya
    text_c = "kucing bermain di taman"              # tidak relevan
    
    vec_a = embedder.embed_text(text_a)
    vec_b = embedder.embed_text(text_b)
    vec_c = embedder.embed_text(text_c)
    
    # Karena normalized, dot product = cosine similarity
    sim_ab = float(np.dot(vec_a, vec_b))
    sim_ac = float(np.dot(vec_a, vec_c))
    
    print(f"    '{text_a}' vs '{text_b}' → similarity: {sim_ab:.4f} (seharusnya tinggi)")
    print(f"    '{text_a}' vs '{text_c}' → similarity: {sim_ac:.4f} (seharusnya rendah)")
    
    assert sim_ab > sim_ac, "ERROR: Similarity teks mirip seharusnya lebih tinggi!"
    print("    ✓ Test passed!")