"""
query.py — API endpoint untuk Q&A dan analisis kontrak

Endpoints:
- POST /query/ask        — Tanya pertanyaan tentang kontrak
- POST /query/summarize  — Buat ringkasan kontrak
- POST /query/explain    — Jelaskan pasal tertentu
- POST /query/risks      — Deteksi risiko dalam kontrak
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from llm.qa_chain import LegalDocRAGChain
from config.settings import settings

logger = logging.getLogger(__name__)

# Create router
query_router = APIRouter(
    prefix="/query",
    tags=["query"],
)


# ─── Request/Response Models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body untuk ask endpoint."""
    question: str = Field(..., min_length=3, max_length=500)
    filter_source: Optional[str] = Field(None, description="Filter berdasarkan nama file tertentu")
    include_sources: bool = Field(False, description="Return chunks yang digunakan?")


class SourceInfo(BaseModel):
    """Info tentang sumber (chunk) yang digunakan untuk jawaban."""
    chunk_id: str
    pasal_number: int
    pasal_title: str
    source_file: str
    relevance_score: float
    text_preview: str = Field(..., description="200 karakter pertama")


class AskResponse(BaseModel):
    """Response untuk ask endpoint."""
    answer: str
    sources: Optional[list[SourceInfo]] = None


class SummarizeRequest(BaseModel):
    """Request body untuk summarize endpoint."""
    filter_source: Optional[str] = None
    focus_areas: Optional[list[str]] = Field(
        None,
        description="Area yang ingin di-highlight, misal: ['pembayaran', 'terminasi']"
    )


class SummarizeResponse(BaseModel):
    """Response untuk summarize endpoint."""
    summary: str
    source_file: Optional[str] = None


class ExplainRequest(BaseModel):
    """Request body untuk explain endpoint."""
    pasal_number: int = Field(..., ge=1, le=1000)
    filter_source: Optional[str] = None


class ExplainResponse(BaseModel):
    """Response untuk explain endpoint."""
    pasal_number: int
    pasal_title: str
    explanation: str
    source_file: str


class RisksRequest(BaseModel):
    """Request body untuk risks endpoint."""
    filter_source: Optional[str] = None
    use_llm: bool = Field(
        False,
        description="Gunakan LLM untuk deteksi (lebih lambat tapi bisa lebih nuanced)"
    )


class RiskInfo(BaseModel):
    """Info tentang satu risiko yang terdeteksi."""
    name: str
    risk_level: str  # "RED" atau "YELLOW"
    pasal_number: int
    pasal_title: str
    description: str
    advice: str
    similarity_score: Optional[float] = None


class RisksResponse(BaseModel):
    """Response untuk risks endpoint."""
    overall_risk_level: str  # "RED", "YELLOW", "GREEN"
    risk_summary: str
    red_flags: list[RiskInfo]
    yellow_flags: list[RiskInfo]


# ─── Global RAG Chain Instance ───────────────────────────────────────────────

# Inisialisasi RAG chain satu kali saat startup
# (akan di-load saat pertama kali endpoint dipanggil - lazy loading)
_rag_chain: Optional[LegalDocRAGChain] = None


def get_rag_chain() -> LegalDocRAGChain:
    """
    Get atau create RAG chain instance (singleton pattern).
    
    Lazy loading: chain baru di-init saat pertama kali dipakai,
    bukan saat server startup. Ini menghemat waktu startup.
    """
    global _rag_chain
    
    if _rag_chain is None:
        logger.info("Initializing RAG chain (first time)...")
        try:
            _rag_chain = LegalDocRAGChain()
            logger.info("✓ RAG chain initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"RAG chain initialization failed: {str(e)}",
            )
    
    return _rag_chain


# ─── Endpoints ───────────────────────────────────────────────────────────────

@query_router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Tanya pertanyaan tentang kontrak dan dapatkan jawaban berbasis RAG.
    
    Example request:
    ```json
    {
        "question": "Berapa denda jika terlambat bayar?",
        "filter_source": "Kontrak_Sewa.pdf",
        "include_sources": true
    }
    ```
    
    Example response:
    ```json
    {
        "answer": "Menurut Pasal 8, keterlambatan pembayaran dikenakan denda 2% per bulan...",
        "sources": [
            {
                "chunk_id": "kontrak_sewa_pasal_8",
                "pasal_number": 8,
                "pasal_title": "Denda",
                "source_file": "Kontrak_Sewa.pdf",
                "relevance_score": 0.87,
                "text_preview": "Keterlambatan pembayaran dikenakan denda..."
            }
        ]
    }
    ```
    """
    try:
        chain = get_rag_chain()
        
        result = chain.ask(
            question=request.question,
            filter_source=request.filter_source,
            return_sources=request.include_sources,
        )
        
        response_data = {"answer": result["answer"]}
        
        # Format sources jika diminta
        if request.include_sources and result.get("sources"):
            response_data["sources"] = [
                SourceInfo(
                    chunk_id=src.chunk_id,
                    pasal_number=src.pasal_number,
                    pasal_title=src.pasal_title,
                    source_file=src.source_file,
                    relevance_score=src.score,
                    text_preview=src.text[:200].replace("\n", " ") + "...",
                )
                for src in result["sources"]
            ]
        
        return AskResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Ask endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}",
        )


@query_router.post("/summarize", response_model=SummarizeResponse)
def summarize_contract(request: SummarizeRequest):
    """
    Buat ringkasan kontrak.
    
    Example request:
    ```json
    {
        "filter_source": "Kontrak_Sewa.pdf",
        "focus_areas": ["pembayaran", "terminasi", "risiko"]
    }
    ```
    """
    try:
        chain = get_rag_chain()
        
        summary = chain.summarize(
            filter_source=request.filter_source,
            focus_areas=request.focus_areas,
        )
        
        return SummarizeResponse(
            summary=summary,
            source_file=request.filter_source,
        )
    
    except Exception as e:
        logger.error(f"Summarize endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating summary: {str(e)}",
        )


@query_router.post("/explain", response_model=ExplainResponse)
def explain_clause(request: ExplainRequest):
    """
    Jelaskan satu pasal tertentu dalam bahasa awam.
    
    Example request:
    ```json
    {
        "pasal_number": 5,
        "filter_source": "Kontrak_Sewa.pdf"
    }
    ```
    """
    try:
        chain = get_rag_chain()
        
        explanation = chain.explain_clause(
            pasal_number=request.pasal_number,
            filter_source=request.filter_source,
        )
        
        # Jika tidak ditemukan, explanation akan berisi pesan error
        if "tidak ditemukan" in explanation.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pasal {request.pasal_number} tidak ditemukan dalam kontrak",
            )
        
        # Cari info pasal dari searcher (simplified)
        # Untuk production bisa query langsung ke Qdrant
        return ExplainResponse(
            pasal_number=request.pasal_number,
            pasal_title=f"Pasal {request.pasal_number}",  # Simplified
            explanation=explanation,
            source_file=request.filter_source or "kontrak",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explain endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error explaining clause: {str(e)}",
        )


@query_router.post("/risks", response_model=RisksResponse)
def detect_risks(request: RisksRequest):
    """
    Deteksi risiko dalam kontrak.
    
    Dua mode:
    1. use_llm=False → Pakai risk_detector.py (similarity-based, cepat)
    2. use_llm=True  → Pakai LLM (lebih nuanced, lambat)
    
    Example request:
    ```json
    {
        "filter_source": "Kontrak_Sewa.pdf",
        "use_llm": false
    }
    ```
    """
    try:
        if request.use_llm:
            # Deteksi pakai LLM
            chain = get_rag_chain()
            analysis = chain.detect_risks_interactive(
                filter_source=request.filter_source,
            )
            
            # Parse LLM output ke structured format
            # (simplified — production bisa pakai prompt yang return JSON)
            return RisksResponse(
                overall_risk_level="YELLOW",  # Default
                risk_summary=analysis,
                red_flags=[],
                yellow_flags=[],
            )
        
        else:
            # Deteksi pakai similarity (dari risk_detector.py)
            from risk_detector.detector import RiskDetector
            from ingestion.chunker import chunk_pdf
            from pathlib import Path
            
            detector = RiskDetector()
            
            # Simplified: asumsi file ada di upload_dir
            upload_dir = Path(settings.upload_dir)
            file_path = upload_dir / request.filter_source if request.filter_source else None
            
            if not file_path or not file_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File '{request.filter_source}' tidak ditemukan",
                )
            
            chunks = chunk_pdf(file_path)
            report = detector.analyze_document(chunks, source_file=request.filter_source)
            
            # Convert ke response format
            red_flags = [
                RiskInfo(
                    name=r.detected_risks[0]["name"] if r.detected_risks else "Unknown",
                    risk_level="RED",
                    pasal_number=r.pasal_number,
                    pasal_title=r.pasal_title,
                    description=r.detected_risks[0]["description"] if r.detected_risks else "",
                    advice=r.detected_risks[0]["advice"] if r.detected_risks else "",
                    similarity_score=r.max_similarity,
                )
                for r in report.red_flags
            ]
            
            yellow_flags = [
                RiskInfo(
                    name=r.detected_risks[0]["name"] if r.detected_risks else "Unknown",
                    risk_level="YELLOW",
                    pasal_number=r.pasal_number,
                    pasal_title=r.pasal_title,
                    description=r.detected_risks[0]["description"] if r.detected_risks else "",
                    advice=r.detected_risks[0]["advice"] if r.detected_risks else "",
                    similarity_score=r.max_similarity,
                )
                for r in report.yellow_flags
            ]
            
            return RisksResponse(
                overall_risk_level=report.overall_risk_level,
                risk_summary=report.risk_summary,
                red_flags=red_flags,
                yellow_flags=yellow_flags,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risks endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting risks: {str(e)}",
        )


# ─── Health Check ────────────────────────────────────────────────────────────

@query_router.get("/health")
def query_health():
    """Health check untuk query service."""
    try:
        # Cek apakah RAG chain bisa di-init
        chain = get_rag_chain()
        
        return {
            "status": "healthy",
            "rag_chain_initialized": _rag_chain is not None,
            "retrieval_top_k": settings.retrieval_top_k,
            "llm_model": settings.groq_model,
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }