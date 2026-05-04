"""
prompts.py — Template prompt yang bagus untuk Q&A dan summary kontrak

Apa itu Prompt?
  Prompt adalah "instruksi" yang kita berikan ke LLM agar menghasilkan output
  yang kita inginkan. Prompt yang bagus = output yang bagus.
  
  Analogi: Prompt seperti brief ke desainer. Makin jelas brief-nya, makin bagus hasilnya.

Komponen Prompt yang Baik:
  1. ROLE — Siapa peran LLM? (legal assistant, contract reviewer, dll.)
  2. TASK — Apa yang harus dilakukan? (jawab pertanyaan, summarize, detect risk, dll.)
  3. CONTEXT — Informasi apa yang diberikan? (pasal-pasal kontrak dari retrieval)
  4. CONSTRAINTS — Batasan apa? (jangan halusinasi, jawab berdasarkan context saja, dll.)
  5. FORMAT — Format output seperti apa? (bullet points, paragraf, JSON, dll.)

Prompt Engineering untuk Legal Domain:
  - Tekankan: "Jawab HANYA berdasarkan pasal yang diberikan"
  - Minta LLM kutip nomor pasal sebagai referensi
  - Jika tidak ada jawabannya di context, suruh bilang "Tidak ditemukan"
  - Hindari halusinasi dengan explicit constraint
"""

from typing import Optional


# ─── System Prompts ──────────────────────────────────────────────────────────

LEGAL_ASSISTANT_SYSTEM_PROMPT = """Kamu adalah asisten hukum AI yang membantu memahami kontrak dalam bahasa Indonesia dan Inggris.

TUGAS UTAMA:
- Menjawab pertanyaan tentang kontrak berdasarkan pasal-pasal yang diberikan
- Menjelaskan klausul hukum dalam bahasa yang mudah dipahami non-lawyer
- Mendeteksi risiko dan memberikan saran

ATURAN PENTING:
1. Jawab HANYA berdasarkan pasal yang diberikan di CONTEXT
2. Jika jawabannya tidak ada di context, katakan: "Informasi ini tidak ditemukan dalam kontrak yang diberikan."
3. SELALU kutip nomor pasal sebagai referensi (misal: "Menurut Pasal 5...")
4. Jelaskan istilah hukum yang rumit dalam bahasa sederhana
5. Jika ada klausul yang berpotensi merugikan, WAJIB beritahu user
6. JANGAN menambahkan informasi dari pengetahuan umum jika tidak relevan dengan kontrak ini

GAYA KOMUNIKASI:
- Ramah dan jelas
- Bahasa Indonesia yang baik tapi tidak kaku
- Gunakan bullet points untuk daftar hal-hal
- Highlight risiko dengan tegas tapi tidak menakut-nakuti"""


CONTRACT_SUMMARIZER_SYSTEM_PROMPT = """Kamu adalah AI yang ahli dalam merangkum kontrak hukum.

TUGAS:
Buat ringkasan kontrak yang mencakup:
- Pihak-pihak yang terlibat
- Objek perjanjian (apa yang diperjanjikan)
- Hak dan kewajiban masing-masing pihak
- Durasi / jangka waktu
- Syarat pembayaran
- Klausul terminasi (pengakhiran)
- Risiko / red flags yang perlu diperhatikan

FORMAT:
Gunakan struktur yang jelas dengan heading dan bullet points.

ATURAN:
- Ringkas tapi tidak kehilangan detail penting
- Highlight klausul yang unusual atau berisiko
- Gunakan bahasa yang bisa dipahami orang awam"""


RISK_ANALYZER_SYSTEM_PROMPT = """Kamu adalah legal risk analyst yang menganalisis klausul kontrak.

TUGAS:
Identifikasi dan jelaskan risiko dalam kontrak, terutama:
- Indemnifikasi sepihak
- Auto-renewal tanpa notifikasi memadai
- Perubahan sepihak
- Pembatasan kewajiban yang tidak seimbang
- Yurisdiksi asing
- Klausul non-kompetisi berlebihan
- Denda atau penalti yang tidak wajar

OUTPUT:
Untuk setiap risiko yang ditemukan:
1. Nama risiko
2. Di pasal mana (kutip nomor pasal)
3. Kenapa berbahaya
4. Saran untuk negosiasi / mitigasi

Jika tidak ada risiko signifikan, katakan: "Kontrak ini tampak cukup seimbang, tidak ditemukan red flag yang mencolok."
"""


# ─── Prompt Templates ────────────────────────────────────────────────────────

def build_qa_prompt(
    question: str,
    context: str,
    include_risk_warning: bool = True,
) -> str:
    """
    Buat prompt untuk Q&A tentang kontrak.
    
    Args:
        question           : Pertanyaan user
        context            : Pasal-pasal relevan dari retrieval (formatted)
        include_risk_warning: Tambahkan perintah untuk deteksi risiko
        
    Returns:
        String prompt siap dikirim ke LLM
    """
    risk_instruction = ""
    if include_risk_warning:
        risk_instruction = """
PENTING: Jika dalam menjawab pertanyaan ini kamu menemukan klausul yang berpotensi merugikan user, WAJIB beritahu secara eksplisit dengan contoh seperti:
"⚠️ PERHATIAN: Pasal X mengandung klausul [nama risiko] yang bisa merugikan karena [alasan]. Sebaiknya [saran]."
"""
    
    prompt = f"""Berdasarkan pasal-pasal kontrak berikut, jawab pertanyaan user.

CONTEXT (Pasal-pasal Relevan):
{context}

PERTANYAAN USER:
{question}

{risk_instruction}

JAWABAN:"""
    
    return prompt


def build_summary_prompt(
    context: str,
    focus_areas: Optional[list[str]] = None,
) -> str:
    """
    Buat prompt untuk summary kontrak.
    
    Args:
        context    : Semua pasal kontrak (atau pasal-pasal penting)
        focus_areas: Area spesifik yang ingin di-highlight (opsional)
                     Contoh: ["pembayaran", "terminasi", "risiko"]
    
    Returns:
        String prompt untuk summarization
    """
    focus_instruction = ""
    if focus_areas:
        areas_str = ", ".join(focus_areas)
        focus_instruction = f"\nBerikan perhatian khusus pada: {areas_str}"
    
    prompt = f"""Buatkan ringkasan kontrak berdasarkan pasal-pasal berikut.{focus_instruction}

PASAL-PASAL KONTRAK:
{context}

INSTRUKSI:
Buat ringkasan yang mencakup:
1. **Pihak-pihak**: Siapa saja yang terlibat dan kapasitasnya
2. **Objek Perjanjian**: Apa yang diperjanjikan (sewa, jual-beli, layanan, dll.)
3. **Hak & Kewajiban**: Breakdown hak dan kewajiban masing-masing pihak
4. **Pembayaran**: Nominal, cara bayar, jatuh tempo, denda keterlambatan
5. **Durasi**: Jangka waktu berlaku, perpanjangan, terminasi
6. **Risiko**: Klausul yang perlu diwaspadai (red flags)

Format: Gunakan heading (##) dan bullet points untuk struktur yang jelas.

RINGKASAN KONTRAK:"""
    
    return prompt


def build_comparison_prompt(
    context_a: str,
    context_b: str,
    comparison_aspects: Optional[list[str]] = None,
) -> str:
    """
    Buat prompt untuk membandingkan dua kontrak.
    
    Args:
        context_a         : Pasal-pasal dari kontrak A
        context_b         : Pasal-pasal dari kontrak B
        comparison_aspects: Aspek yang ingin dibandingkan (opsional)
                            Contoh: ["pembayaran", "terminasi", "tanggung jawab"]
    
    Returns:
        String prompt untuk comparison
    """
    aspects_instruction = ""
    if comparison_aspects:
        aspects_str = ", ".join(comparison_aspects)
        aspects_instruction = f"\nFokus perbandingan pada aspek: {aspects_str}"
    
    prompt = f"""Bandingkan dua kontrak berikut.{aspects_instruction}

KONTRAK A:
{context_a}

KONTRAK B:
{context_b}

INSTRUKSI:
Buat tabel perbandingan yang mencakup:
- Pihak yang terlibat
- Objek perjanjian
- Durasi
- Syarat pembayaran
- Klausul terminasi
- Risiko / red flags

Untuk setiap aspek, sebutkan:
- Apa yang tercantum di Kontrak A
- Apa yang tercantum di Kontrak B
- Mana yang lebih menguntungkan / lebih aman

PERBANDINGAN:"""
    
    return prompt


def build_clause_explanation_prompt(
    clause_text: str,
    clause_reference: str = "klausul ini",
) -> str:
    """
    Buat prompt untuk menjelaskan klausul tertentu dalam bahasa awam.
    
    Args:
        clause_text     : Teks klausul yang ingin dijelaskan
        clause_reference: Referensi klausul (misal: "Pasal 5 ayat 2")
    
    Returns:
        String prompt untuk explanation
    """
    prompt = f"""Jelaskan {clause_reference} dalam bahasa yang mudah dipahami orang awam.

KLAUSUL:
{clause_text}

INSTRUKSI:
1. Jelaskan maksud klausul ini dalam bahasa sehari-hari
2. Berikan contoh konkret jika memungkinkan
3. Jelaskan implikasi / konsekuensi dari klausul ini
4. Jika ada istilah hukum (legalese), jelaskan artinya
5. Beritahu jika klausul ini standar atau unusual
6. Warning jika klausul ini bisa merugikan salah satu pihak

PENJELASAN:"""
    
    return prompt


# ─── Prompt Helpers ──────────────────────────────────────────────────────────

def format_context_from_results(search_results: list) -> str:
    """
    Format search results menjadi context string untuk prompt.
    
    Args:
        search_results: List of SearchResult dari searcher.py
        
    Returns:
        Formatted context string
    """
    if not search_results:
        return "Tidak ada pasal yang relevan ditemukan."
    
    formatted_chunks = [r.format_for_llm() for r in search_results]
    return "\n\n".join(formatted_chunks)


def truncate_context(context: str, max_length: int = 4000) -> str:
    """
    Truncate context jika terlalu panjang.
    
    Kenapa perlu truncate?
    LLM punya batasan panjang input (context window). Jika context terlalu panjang,
    request akan gagal atau biaya API membengkak.
    
    Args:
        context   : Context string
        max_length: Maksimum panjang karakter
        
    Returns:
        Truncated context (jika perlu)
    """
    if len(context) <= max_length:
        return context
    
    truncated = context[:max_length]
    truncated += "\n\n[... Context dipotong karena terlalu panjang ...]"
    return truncated


# ─── Prompt Variations (untuk A/B Testing) ───────────────────────────────────

def build_qa_prompt_concise(question: str, context: str) -> str:
    """Variasi prompt Q&A yang lebih ringkas (untuk model kecil / cepat)."""
    return f"""Context: {context}

Question: {question}

Instructions: Answer based only on the context. Cite article numbers. If not found, say "Not found in contract."

Answer:"""


def build_qa_prompt_detailed(question: str, context: str) -> str:
    """Variasi prompt Q&A yang lebih detail (untuk model besar / akurat)."""
    return f"""You are a legal contract expert helping a user understand their contract.

CONTEXT (Relevant Contract Clauses):
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer based STRICTLY on the provided clauses
2. Always cite the article/clause number when referencing information
3. Explain legal terms in plain language
4. If the answer is not in the context, clearly state: "This information is not found in the provided contract clauses."
5. If you notice any risky clauses while answering, warn the user
6. Structure your answer clearly with paragraphs or bullet points

ANSWER:"""


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test prompt building"""
    print("="*60)
    print("TEST PROMPT TEMPLATES")
    print("="*60)
    
    # Mock search results
    mock_context = """---
[Pasal 5 - Pembayaran]
(Sumber: Kontrak_Sewa.pdf | Relevance: 0.87)

Penyewa wajib membayar sewa sebesar Rp 10.000.000 per bulan, 
dibayarkan paling lambat tanggal 5 setiap bulannya.
---

---
[Pasal 8 - Denda]
(Sumber: Kontrak_Sewa.pdf | Relevance: 0.76)

Keterlambatan pembayaran dikenakan denda 2% per bulan dari 
jumlah yang tertunggak.
---"""
    
    print("\n[1] Q&A Prompt:")
    print("-" * 60)
    qa_prompt = build_qa_prompt(
        question="Berapa denda jika terlambat bayar?",
        context=mock_context,
    )
    print(qa_prompt[:400] + "...")
    
    print("\n[2] Summary Prompt:")
    print("-" * 60)
    summary_prompt = build_summary_prompt(
        context=mock_context,
        focus_areas=["pembayaran", "denda"],
    )
    print(summary_prompt[:400] + "...")
    
    print("\n[3] Clause Explanation Prompt:")
    print("-" * 60)
    explain_prompt = build_clause_explanation_prompt(
        clause_text="Keterlambatan pembayaran dikenakan denda 2% per bulan",
        clause_reference="Pasal 8 tentang Denda",
    )
    print(explain_prompt[:400] + "...")
    
    print("\n" + "="*60)
    print("✓ Prompt templates siap digunakan!")