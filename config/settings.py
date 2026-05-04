"""
settings.py — Konfigurasi aplikasi dari environment variables

Kenapa pakai config file?
  - Centralize semua konfigurasi di satu tempat
  - Mudah di-test (bisa mock settings)
  - Type-safe dengan Pydantic
  - Auto-validasi (error jika config salah)
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded dari environment variables.
    
    Pydantic BaseSettings otomatis:
    - Load dari env vars
    - Validasi tipe data
    - Provide default values
    - Case-insensitive (APP_NAME = app_name)
    """
    
    # ─── General ────────────────────────────────────────────────────────────
    app_name: str = "LegalDoc RAG API"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # ─── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "legaldoc_chunks"
    
    # ─── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = "nlpaueb/legal-bert-base-uncased"
    embedding_dimension: int = 768
    
    # ─── LLM (Groq) ─────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 2000
    
    # ─── RAG ────────────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.5
    max_context_length: int = 4000
    
    # ─── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True  # Auto-reload saat development
    
    # CORS — allow frontend dari origin tertentu
    cors_origins: list[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # ─── Upload ─────────────────────────────────────────────────────────────
    upload_dir: str = "data/raw"
    max_upload_size_mb: int = 20
    allowed_extensions: set[str] = {".pdf", ".docx"}
    
    # ─── Risk Detection ─────────────────────────────────────────────────────
    risk_red_threshold: float = 0.72
    risk_yellow_threshold: float = 0.55
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# ─── Singleton Instance ──────────────────────────────────────────────────────

# Buat satu instance global yang bisa di-import dari mana saja
settings = Settings()


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_settings() -> Settings:
    """
    Dependency injection untuk FastAPI.
    
    Usage di FastAPI routes:
        from fastapi import Depends
        from config.settings import get_settings, Settings
        
        @app.get("/config")
        def show_config(settings: Settings = Depends(get_settings)):
            return {"app_name": settings.app_name}
    """
    return settings


def validate_settings() -> dict:
    """
    Validasi semua settings dan return report.
    Berguna untuk debugging saat deployment.
    
    Returns:
        Dict dengan status validasi
    """
    issues = []
    
    # Cek Groq API key
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        issues.append("GROQ_API_KEY belum di-set atau masih placeholder")
    
    # Cek upload directory
    from pathlib import Path
    upload_path = Path(settings.upload_dir)
    if not upload_path.exists():
        upload_path.mkdir(parents=True, exist_ok=True)
        issues.append(f"Upload directory '{settings.upload_dir}' tidak ada, sudah dibuat otomatis")
    
    # Cek retrieval thresholds
    if settings.retrieval_score_threshold > 0.9:
        issues.append(
            f"retrieval_score_threshold sangat tinggi ({settings.retrieval_score_threshold}), "
            "mungkin tidak ada hasil yang ditemukan. Rekomendasi: 0.3 - 0.6"
        )
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "settings_summary": {
            "qdrant": f"{settings.qdrant_host}:{settings.qdrant_port}",
            "model": settings.embedding_model,
            "llm": settings.groq_model,
            "retrieval_top_k": settings.retrieval_top_k,
        }
    }


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test settings loading"""
    print("="*60)
    print("SETTINGS VALIDATION")
    print("="*60)
    
    validation = validate_settings()
    
    print(f"\nStatus: {'✓ Valid' if validation['valid'] else '✗ Ada Issues'}")
    
    if validation['issues']:
        print("\nIssues:")
        for issue in validation['issues']:
            print(f"  ⚠️  {issue}")
    
    print("\nSettings Summary:")
    for key, value in validation['settings_summary'].items():
        print(f"  {key:20s}: {value}")
    
    print("\nFull Settings:")
    print(f"  App Name         : {settings.app_name}")
    print(f"  Version          : {settings.app_version}")
    print(f"  Qdrant           : {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"  Collection       : {settings.qdrant_collection}")
    print(f"  Embedding Model  : {settings.embedding_model}")
    print(f"  LLM Model        : {settings.groq_model}")
    print(f"  API Host:Port    : {settings.api_host}:{settings.api_port}")
    print(f"  Upload Dir       : {settings.upload_dir}")
    print(f"  Max Upload Size  : {settings.max_upload_size_mb} MB")
    
    print("\n" + "="*60)