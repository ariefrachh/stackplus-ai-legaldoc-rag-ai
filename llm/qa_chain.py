"""
qa_chain.py — RAG Chain: Retrieval + LLM untuk menjawab pertanyaan kontrak

Apa itu RAG Chain?
  RAG = Retrieval-Augmented Generation
  Chain = rangkaian proses dari input sampai output
  
  Alur:
  1. User tanya: "Apa risiko klausul indemnifikasi?"
  2. RETRIEVAL: Cari 5 pasal paling relevan dari Qdrant
  3. PROMPT: Gabungkan pasal-pasal + pertanyaan jadi prompt
  4. LLM: Kirim prompt ke LLM (Llama 3 via Groq)
  5. RESPONSE: LLM jawab berdasarkan pasal-pasal tersebut
  6. Return jawaban ke user

Kenapa pakai RAG, bukan langsung tanya LLM?
  - LLM tanpa context bisa halusinasi (ngaco, jawab asal)
  - LLM tidak tahu isi kontrak spesifik kamu
  - Dengan RAG, jawaban LLM berbasis dokumen nyata → akurat & bisa di-audit
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from llm.query_parser import parse_legal_query
from retrieval.searcher import LegalDocSearcher, SearchResult
from llm.prompts import (
    build_qa_prompt,
    build_summary_prompt,
    build_clause_explanation_prompt,
    LEGAL_ASSISTANT_SYSTEM_PROMPT,
    CONTRACT_SUMMARIZER_SYSTEM_PROMPT,
)

# Load environment variables dari .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── LLM Client ──────────────────────────────────────────────────────────────

class GroqLLMClient:
    """
    Client untuk Groq API (Llama 3).
    
    Groq adalah inference engine yang super cepat untuk LLM.
    Mereka support Llama 3 70B dengan kecepatan ~300 tokens/detik.
    
    API key gratis bisa didapat di: https://console.groq.com
    
    Usage:
        client = GroqLLMClient(api_key="your_key")
        response = client.chat("Halo, apa kabar?")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",  # Model terbaru Groq
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """
        Inisialisasi Groq client.
        
        Args:
            api_key    : Groq API key (ambil dari env jika None)
            model      : Nama model Groq (default: llama-3.3-70b-versatile)
            temperature: Kreativitas model (0.0 = deterministik, 1.0 = kreatif)
                         Untuk legal, pakai rendah (0.0-0.2)
            max_tokens : Maksimum panjang response
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key tidak ditemukan. "
                "Set di .env: GROQ_API_KEY=your_key atau pass langsung ke constructor."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Groq client library."""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✓ Groq client initialized (model={self.model})")
        except ImportError:
            raise ImportError(
                "Library 'groq' tidak terinstall. "
                "Install dengan: pip install groq"
            )
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Kirim message ke LLM dan dapatkan response.
        
        Args:
            user_message : Prompt user (bisa dari build_qa_prompt, dll.)
            system_prompt: System prompt (opsional, override default)
            temperature  : Override temperature default (opsional)
            
        Returns:
            String response dari LLM
        """
        messages = []
        
        # System prompt (role definition)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        # User message
        messages.append({
            "role": "user",
            "content": user_message,
        })
        
        # API call ke Groq
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
            )
            
            response_text = completion.choices[0].message.content
            return response_text
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


# ─── RAG Chain ───────────────────────────────────────────────────────────────

class LegalDocRAGChain:
    """
    RAG Chain untuk Q&A kontrak hukum.
    
    Ini adalah "otak" dari sistem LegalDoc RAG. Menggabungkan:
    - Searcher (retrieval)
    - Prompt builder
    - LLM (generation)
    
    Usage:
        chain = LegalDocRAGChain()
        answer = chain.ask("Apa risiko klausul indemnifikasi?")
        print(answer)
    """
    
    def __init__(
        self,
        searcher: Optional[LegalDocSearcher] = None,
        llm_client: Optional[GroqLLMClient] = None,
        retrieval_top_k: int = 5,
    ):
        """
        Inisialisasi RAG chain.
        
        Args:
            searcher       : LegalDocSearcher instance (dibuat baru jika None)
            llm_client     : GroqLLMClient instance (dibuat baru jika None)
            retrieval_top_k: Jumlah chunks yang di-retrieve per query
        """
        # Initialize searcher
        if searcher is None:
            logger.info("Inisialisasi searcher untuk RAG chain...")
            from retrieval.searcher import get_default_searcher
            self.searcher = get_default_searcher()
        else:
            self.searcher = searcher
        
        # Initialize LLM
        if llm_client is None:
            logger.info("Inisialisasi Groq LLM client...")
            self.llm = GroqLLMClient()
        else:
            self.llm = llm_client
        
        self.retrieval_top_k = retrieval_top_k
        logger.info("✓ RAG Chain siap digunakan!")
    
    def ask(
        self,
        question: str,
        filter_source: Optional[str] = None,
        include_risk_warning: bool = True,
        return_sources: bool = False,
    ) -> dict:

        logger.info(f"Processing question: '{question[:60]}...'")

        # 🔥 NORMALISASI SOURCE (WAJIB)
        if filter_source:
            filter_source = filter_source.strip()

        # 🔥 Step 0: Parse query (pasal & ayat)
        try:
            from llm.query_parser import parse_legal_query
            pasal, ayat = parse_legal_query(question)
        except Exception as e:
            logger.warning(f"Query parsing failed: {e}")
            pasal, ayat = None, None

        # 🔥 Step 0b: DETECT QUERY TYPE
        lower_q = question.lower()

        is_preamble_query = any(k in lower_q for k in [
            "judul kontrak",
            "nama kontrak",
            "perjanjian ini",
            "judul perjanjian",
            "pihak pertama",
            "pihak kedua",
            "para pihak",
            "siapa yang menyewakan",
            "siapa penyewa",
        ])

        # 🔥 QUERY BOOST (RISK CLAUSE)
        if any(k in lower_q for k in [
            "klausul berbahaya",
            "risiko kontrak",
            "klausul merugikan"
        ]):
            logger.info("Applying query boost for risk-related query")
            question += " penalti denda kewajiban ganti rugi risiko pelanggaran kontrak"

        if is_preamble_query:
            logger.info("Detected preamble-related query → forcing pasal=0")
            pasal = 0
            ayat = None

        # 🔥 Step 1: Retrieval (threshold diturunkan)
        search_results = self.searcher.search(
            query=question,
            top_k=self.retrieval_top_k,
            score_threshold=0.3,  # 🔥 lebih longgar
            filter_source=filter_source,
            filter_pasal=pasal,
            filter_ayat=ayat,
        )

        # 🔥 DEBUG RETRIEVAL
        print("\n🔥 ===== RETRIEVED CONTEXT (STEP 1) =====")
        for i, r in enumerate(search_results):
            print(f"\n--- Chunk {i+1} ---")
            print(r.text[:300])
            print("Metadata:", r.metadata)
        print("🔥 ===== END CONTEXT =====\n")

        # 🔥 Step 1b: fallback tanpa ayat
        if not search_results and pasal is not None:
            logger.info("Fallback: retry search tanpa filter ayat")
            search_results = self.searcher.search(
                query=question,
                top_k=self.retrieval_top_k,
                score_threshold=0.2,
                filter_source=filter_source,
                filter_pasal=pasal,
                filter_ayat=None,
            )

        # 🔥 Step 1c: fallback khusus PREAMBLE
        if not search_results and is_preamble_query:
            logger.info("Fallback: force ambil preamble (pasal=0, tanpa threshold)")
            search_results = self.searcher.search(
                query=question,
                top_k=5,
                score_threshold=0.0,
                filter_source=filter_source,
                filter_pasal=0,
            )

        # 🔥 Step 1d: fallback tanpa filter_source (KRUSIAL)
        if not search_results and filter_source:
            logger.info("Fallback: retry WITHOUT source filter")
            search_results = self.searcher.search(
                query=question,
                top_k=self.retrieval_top_k,
                score_threshold=0.3,
                filter_source=None,
            )

        # 🔥 Step 1e: fallback total (semantic only)
        if not search_results:
            logger.info("Fallback: retry search tanpa filter pasal/ayat")
            search_results = self.searcher.search(
                query=question,
                top_k=self.retrieval_top_k,
                score_threshold=0.3,
                filter_source=filter_source,
            )

        # ❌ Jika tetap kosong
        if not search_results:
            logger.warning("Tidak ada chunks relevan ditemukan untuk query ini")
            return {
                "answer": (
                    "Maaf, saya tidak menemukan informasi yang relevan "
                    "dalam kontrak untuk menjawab pertanyaan ini."
                ),
                "sources": [] if return_sources else None,
            }

        # 🔥 DEBUG FINAL CONTEXT
        print("\n🔥 ===== FINAL CONTEXT KE LLM =====")
        for i, r in enumerate(search_results):
            print(f"\n--- Final Chunk {i+1} ---")
            print(r.text[:300])
        print("🔥 ===== END FINAL =====\n")

        # 🔥 Step 2: Build prompt
        context = "\n\n".join([r.format_for_llm() for r in search_results])

        prompt = build_qa_prompt(
            question=question,
            context=context,
            include_risk_warning=include_risk_warning,
        )

        # 🔥 Step 3: Generate ke LLM
        logger.info("Generating answer dengan Groq LLM...")

        answer = self.llm.chat(
            user_message=prompt,
            system_prompt=LEGAL_ASSISTANT_SYSTEM_PROMPT,
        )

        logger.info("✓ Answer generated")

        result = {"answer": answer}

        if return_sources:
            result["sources"] = search_results

        return result
    
    def summarize(
        self,
        filter_source: Optional[str] = None,
        focus_areas: Optional[list[str]] = None,
    ) -> str:
        """
        Buat ringkasan kontrak.
        
        Berbeda dengan ask() yang menjawab pertanyaan spesifik,
        summarize() membuat overview keseluruhan kontrak.
        
        Args:
            filter_source: Summarize hanya file tertentu (opsional)
            focus_areas  : Area yang ingin di-highlight (opsional)
                           Contoh: ["pembayaran", "terminasi"]
        
        Returns:
            String summary dari LLM
        """
        logger.info("Generating contract summary...")
        
        # Untuk summary, kita ambil lebih banyak chunks (top-15)
        # agar mencakup lebih banyak aspek kontrak
        search_results = self.searcher.search(
            query="ringkasan kontrak hak kewajiban pembayaran terminasi",
            top_k=15,
            score_threshold=0.2,
            filter_source=filter_source,
        )
        
        context = "\n\n".join([r.format_for_llm() for r in search_results])
        prompt = build_summary_prompt(context, focus_areas=focus_areas)
        
        summary = self.llm.chat(
            user_message=prompt,
            system_prompt=CONTRACT_SUMMARIZER_SYSTEM_PROMPT,
        )
        
        logger.info("✓ Summary generated")
        return summary
    
    def explain_clause(
        self,
        pasal_number: int,
        filter_source: Optional[str] = None,
    ) -> str:
        """
        Jelaskan satu pasal tertentu dalam bahasa awam.
        
        Args:
            pasal_number : Nomor pasal yang ingin dijelaskan
            filter_source: File tertentu (opsional)
            
        Returns:
            String penjelasan dari LLM
        """
        logger.info(f"Explaining Pasal {pasal_number}...")
        
        # Cari chunks dengan pasal_number yang match
        # (ini simplified — production bisa filter langsung di Qdrant)
        search_results = self.searcher.search(
            query=f"pasal {pasal_number}",
            top_k=5,
            filter_source=filter_source,
        )
        
        # Filter hanya pasal yang exact match
        target_chunk = None
        for r in search_results:
            if r.pasal_number == pasal_number:
                target_chunk = r
                break
        
        if not target_chunk:
            return f"Pasal {pasal_number} tidak ditemukan dalam kontrak."
        
        prompt = build_clause_explanation_prompt(
            clause_text=target_chunk.text,
            clause_reference=f"Pasal {pasal_number} - {target_chunk.pasal_title}",
        )
        
        explanation = self.llm.chat(
            user_message=prompt,
            system_prompt=LEGAL_ASSISTANT_SYSTEM_PROMPT,
        )
        
        logger.info("✓ Explanation generated")
        return explanation
    
    def detect_risks_interactive(
        self,
        filter_source: Optional[str] = None,
    ) -> str:
        """
        Deteksi risiko dalam kontrak menggunakan LLM.
        
        Ini berbeda dengan risk_detector.py yang pakai similarity.
        Di sini, LLM membaca kontrak dan deteksi risiko dengan "pemahaman".
        
        Bisa dipakai sebagai second opinion / sanity check terhadap
        hasil dari risk_detector.py.
        
        Args:
            filter_source: File tertentu (opsional)
            
        Returns:
            String analisis risiko dari LLM
        """
        logger.info("Detecting risks dengan LLM...")
        
        from llm.prompts import RISK_ANALYZER_SYSTEM_PROMPT
        
        # Ambil banyak chunks untuk analisis menyeluruh
        search_results = self.searcher.search(
            query="risiko klausul berbahaya indemnifikasi terminasi liability",
            top_k=20,
            score_threshold=0.2,
            filter_source=filter_source,
        )
        
        context = "\n\n".join([r.format_for_llm() for r in search_results])
        
        prompt = f"""Analisis kontrak berikut untuk mendeteksi klausul yang berisiko.

PASAL-PASAL KONTRAK:
{context}

Identifikasi dan jelaskan setiap risiko yang kamu temukan.

ANALISIS RISIKO:"""
        
        analysis = self.llm.chat(
            user_message=prompt,
            system_prompt=RISK_ANALYZER_SYSTEM_PROMPT,
        )
        
        logger.info("✓ Risk analysis generated")
        return analysis


# ─── Fungsi Helper ──────────────────────────────────────────────────────────────

def get_default_chain() -> LegalDocRAGChain:
    """Shortcut untuk mendapatkan RAG chain dengan konfigurasi default."""
    return LegalDocRAGChain()


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Test RAG chain end-to-end.
    
    Pastikan:
    1. Qdrant jalan: docker-compose up -d qdrant
    2. Data sudah di-ingest: python -m ingestion.vector_store
    3. GROQ_API_KEY sudah di-set di .env
    """
    import sys
    
    print("="*60)
    print("TEST RAG CHAIN")
    print("="*60)
    
    # Cek API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n✗ Error: GROQ_API_KEY belum di-set di .env")
        print("\n1. Daftar gratis di: https://console.groq.com")
        print("2. Buat API key")
        print("3. Tambahkan ke .env: GROQ_API_KEY=gsk_...")
        sys.exit(1)
    
    try:
        print("\n[1] Inisialisasi RAG chain...")
        chain = LegalDocRAGChain()
        
        print("\n[2] Test Q&A...")
        test_questions = [
            "Berapa denda jika terlambat bayar?",
            "Bagaimana cara mengakhiri kontrak lebih awal?",
            "Siapa yang bertanggung jawab jika terjadi kerusakan?",
        ]
        
        for q in test_questions:
            print(f"\n{'─'*60}")
            print(f"Q: {q}")
            print(f"{'─'*60}")
            
            result = chain.ask(q, return_sources=True)
            
            print(f"A: {result['answer']}\n")
            
            if result.get('sources'):
                print("Sources:")
                for i, src in enumerate(result['sources'][:2], 1):
                    print(f"  {i}. [{src.source_file}] Pasal {src.pasal_number} (score={src.score:.3f})")
        
        print("\n[3] Test summary...")
        summary = chain.summarize()
        print(f"\nRingkasan Kontrak:\n{summary[:500]}...\n")
        
        print("="*60)
        print("✓ RAG Chain berfungsi dengan baik!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)