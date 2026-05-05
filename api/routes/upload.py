"""
upload.py — API endpoint untuk upload dan ingest PDF kontrak

Endpoints:
- POST /upload — Upload PDF, trigger chunking + embedding + simpan ke Qdrant
- GET  /files  — List semua file yang sudah di-upload
- DELETE /files/{filename} — Hapus file dari sistem
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, status, BackgroundTasks
from pydantic import BaseModel

from config.settings import settings
from ingestion.chunker import chunk_pdf
from ingestion.embedder import get_default_embedder
from ingestion.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# Create router
upload_router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response model untuk upload endpoint."""
    success: bool
    message: str
    filename: str
    chunks_count: Optional[int] = None
    file_size_bytes: Optional[int] = None


class FileInfo(BaseModel):
    """Info tentang file yang sudah di-upload."""
    filename: str
    size_bytes: int
    uploaded_at: Optional[str] = None
    chunks_count: Optional[int] = None


class FileListResponse(BaseModel):
    """Response untuk list files."""
    files: list[FileInfo]
    total: int


# ─── Helper Functions ────────────────────────────────────────────────────────

def validate_file_upload(file: UploadFile) -> tuple[bool, str]:
    """
    Validasi file yang di-upload.
    
    Returns:
        (is_valid, error_message)
    """
    # Cek extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        return False, f"File extension tidak didukung. Hanya: {', '.join(settings.allowed_extensions)}"
    
    # Cek size (jika tersedia di metadata)
    # Note: file.size tidak selalu tersedia, tergantung client
    # Validasi size yang lebih ketat bisa dilakukan saat membaca file
    
    return True, ""


async def save_uploaded_file(file: UploadFile, destination: Path) -> int:
    """
    Simpan uploaded file ke disk.
    
    Returns:
        File size in bytes
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Read dan save file
    content = await file.read()
    
    # Validasi size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File terlalu besar ({size_mb:.1f} MB). Maksimum: {settings.max_upload_size_mb} MB"
        )
    
    with open(destination, "wb") as f:
        f.write(content)
    
    return len(content)


def ingest_pdf_to_qdrant(pdf_path: Path) -> int:
    logger.info(f"Starting ingestion for: {pdf_path.name}")

    # 🔥 STEP 0: HAPUS DATA LAMA DI QDRANT (PENTING)
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )

        client.delete(
            collection_name=settings.qdrant_collection,
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": "source_file",
                            "match": {"value": pdf_path.name}
                        }
                    ]
                }
            }
        )

        print(f"🧹 Data lama dihapus untuk: {pdf_path.name}")

    except Exception as e:
        print(f"⚠️ Gagal hapus data lama: {e}")

    # 🔥 STEP 1: CHUNKING
    chunks = chunk_pdf(pdf_path)
    print(f"🔥 TOTAL CHUNKS: {len(chunks)}")

    if not chunks:
        raise ValueError(f"Tidak ada chunks dari {pdf_path.name}")

    # 🔥 STEP 2: EMBEDDING
    embedder = get_default_embedder()
    embedded = embedder.embed_chunks(chunks)
    print("🔥 EMBEDDING SELESAI")

    # 🔥 STEP 3: QDRANT
    store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )

    if not store.collection_exists():
        store.setup_collection(vector_dim=embedder.embedding_dim)

    print("🔥 COLLECTION READY")

    upserted = store.upsert(embedded)
    print(f"🔥 UPSERT KE QDRANT: {upserted}")

    logger.info(f"✓ Ingestion selesai: {upserted} chunks")
    return upserted


# ─── Endpoints ───────────────────────────────────────────────────────────────

@upload_router.post("/", response_model=UploadResponse)
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_immediately: bool = False,
):
    """
    Upload PDF dan ingest ke vector database.
    
    Args:
        file               : PDF file yang di-upload
        process_immediately: Jika True, langsung proses (blocking).
                             Jika False, proses di background (return cepat).
    
    Returns:
        UploadResponse dengan status dan info file
    """
    # Validasi file
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )
    
    # Buat path untuk save file
    upload_dir = Path(settings.upload_dir)
    safe_name = file.filename.replace(" ", "_")
    file_path = upload_dir / safe_name
    
    # Cek jika file sudah ada
    # overwrite aja (biar UX enak)
    if file_path.exists():
        logger.info(f"File {file.filename} sudah ada, overwrite...")
    
    try:
        # Save file
        file_size = await save_uploaded_file(file, file_path)
        
        if process_immediately:
            # Proses langsung (blocking)
            chunks_count = ingest_pdf_to_qdrant(file_path)
            
            return UploadResponse(
                success=True,
                message=f"File '{file.filename}' berhasil di-upload dan di-ingest",
                filename=file.filename,
                chunks_count=chunks_count,
                file_size_bytes=file_size,
            )
        else:
            # Proses di background (non-blocking)
            background_tasks.add_task(ingest_pdf_to_qdrant, file_path)
            
            return UploadResponse(
                success=True,
                message=f"File '{file.filename}' berhasil di-upload. Sedang diproses di background.",
                filename=file.filename,
                file_size_bytes=file_size,
            )
    
    except ValueError as e:
        # Error dari ingestion (misal: PDF tidak bisa di-parse)
        # Hapus file yang sudah di-save
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Gagal memproses PDF: {str(e)}",
        )
    
    except Exception as e:
        # Unexpected error
        logger.error(f"Upload error: {e}")
        
        # Cleanup jika ada
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )

@upload_router.get("/files", response_model=FileListResponse)
def list_uploaded_files():
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )

        collection_name = settings.qdrant_collection

        # 🔥 CEK COLLECTION ADA ATAU TIDAK
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            # ✅ belum ada → return kosong
            return FileListResponse(files=[], total=0)

        # 🔥 kalau ada, baru scroll
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True
        )

        points = scroll_result[0]

        filenames = {}

        for point in points:
            source = point.payload.get("source_file")
            if not source:
                continue

            if source not in filenames:
                filenames[source] = {
                    "filename": source,
                    "size_bytes": 0,
                    "chunks_count": 0
                }

            filenames[source]["chunks_count"] += 1

        files_info = [
            FileInfo(
                filename=v["filename"],
                size_bytes=v["size_bytes"],
                chunks_count=v["chunks_count"]
            )
            for v in filenames.values()
        ]

        return FileListResponse(
            files=files_info,
            total=len(files_info)
        )

    except Exception as e:
        logger.error(f"Qdrant list error: {e}")

        # 🔥 fallback aman
        return FileListResponse(files=[], total=0)


@upload_router.delete("/files/{filename}")
def delete_file(filename: str):
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )

    collection_name = settings.qdrant_collection

    try:
        # 🔥 delete dari Qdrant
        client.delete(
            collection_name=collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": "source_file",
                            "match": {
                                "value": filename
                            }
                        }
                    ]
                }
            }
        )

        # 🔥 delete file lokal (optional)
        file_path = Path(settings.upload_dir) / filename
        if file_path.exists():
            file_path.unlink()

        return {
            "success": True,
            "message": f"{filename} berhasil dihapus dari Qdrant & storage"
        }

    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health Check ────────────────────────────────────────────────────────────

@upload_router.get("/health")
def upload_health():
    """Health check untuk upload service."""
    upload_dir = Path(settings.upload_dir)
    
    return {
        "status": "healthy",
        "upload_dir": str(upload_dir),
        "upload_dir_exists": upload_dir.exists(),
        "max_size_mb": settings.max_upload_size_mb,
        "allowed_extensions": list(settings.allowed_extensions),
    }