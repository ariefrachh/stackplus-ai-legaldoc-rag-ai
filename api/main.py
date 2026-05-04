"""
main.py — FastAPI application utama untuk LegalDoc RAG API

Jalankan dengan:
  uvicorn api.main:app --reload
  
Atau:
  python -m api.main
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from api.routes import upload_router, query_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan Events ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager untuk startup dan shutdown events.
    
    Startup:
    - Validasi settings
    - Cek koneksi Qdrant
    - Buat upload directory jika belum ada
    
    Shutdown:
    - Cleanup resources (jika perlu)
    """
    # ─── STARTUP ─────────────────────────────────────────────────────────────
    logger.info("="*60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("="*60)
    
    # Validasi settings
    from config.settings import validate_settings
    validation = validate_settings()
    
    if not validation["valid"]:
        logger.warning("Settings validation issues:")
        for issue in validation["issues"]:
            logger.warning(f"  ⚠️  {issue}")
    
    # Cek Qdrant connection
    try:
        from ingestion.vector_store import QdrantVectorStore
        
        store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        
        if not store.collection_exists():
            logger.warning(
                f"Collection '{settings.qdrant_collection}' belum ada di Qdrant. "
                "Upload PDF pertama akan membuat collection otomatis."
            )
        else:
            info = store.get_collection_info()
            logger.info(
                f"✓ Qdrant connected: {info['vectors_count']} vectors "
                f"in collection '{info['name']}'"
            )
    
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        logger.error("  → Pastikan Qdrant sudah jalan: docker-compose up -d qdrant")
    
    # Buat upload directory
    from pathlib import Path
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Upload directory: {upload_dir}")
    
    logger.info("="*60)
    logger.info(f"API ready at http://{settings.api_host}:{settings.api_port}")
    logger.info(f"Docs at http://{settings.api_host}:{settings.api_port}/docs")
    logger.info("="*60)
    
    yield
    
    # ─── SHUTDOWN ────────────────────────────────────────────────────────────
    logger.info("Shutting down...")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "RAG API untuk analisis kontrak hukum. "
        "Upload PDF, tanya pertanyaan, deteksi risiko, dan buat ringkasan kontrak."
    ),
    lifespan=lifespan,
)


# ─── CORS Middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Exception Handlers ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler untuk catch semua unhandled exceptions.
    
    Ini mencegah server crash dan memberikan error response yang jelas.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.debug else "An error occurred",
        },
    )


# ─── Routers ─────────────────────────────────────────────────────────────────

# Include routers
app.include_router(upload_router)
app.include_router(query_router)


# ─── Root Endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
def root():
    """
    Root endpoint — info tentang API.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs",
        "endpoints": {
            "upload_pdf": "POST /upload/",
            "list_files": "GET /upload/files",
            "ask_question": "POST /query/ask",
            "summarize": "POST /query/summarize",
            "explain_clause": "POST /query/explain",
            "detect_risks": "POST /query/risks",
        },
    }


@app.get("/health", tags=["root"])
def health():
    """
    Health check endpoint untuk monitoring.
    """
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/config", tags=["root"])
def show_config():
    """
    Show current configuration (untuk debugging).
    
    Note: Di production, endpoint ini sebaiknya di-protect atau disabled.
    """
    if not settings.debug:
        return {"error": "Config endpoint disabled in production mode"}
    
    return {
        "qdrant": f"{settings.qdrant_host}:{settings.qdrant_port}",
        "collection": settings.qdrant_collection,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.groq_model,
        "retrieval_top_k": settings.retrieval_top_k,
        "upload_dir": settings.upload_dir,
        "max_upload_size_mb": settings.max_upload_size_mb,
    }


# ─── Run Server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run server via `python -m api.main`
    
    Alternatif (lebih direkomendasikan):
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info",
    )