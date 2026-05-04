# 🗂️ LegalDoc RAG — Panduan Lengkap Week 1-4

> Panduan ini menjelaskan cara setup dan menjalankan LegalDoc RAG dari awal sampai
> API siap digunakan. Mencakup Week 1 (Ingestion), Week 2 (Risk Detection),
> Week 3 (LLM Integration), dan Week 4 (API Layer).

---

## 📋 Daftar Isi

1. [Overview](#overview)
2. [Struktur File Lengkap](#struktur-file-lengkap)
3. [Prasyarat](#prasyarat)
4. [Instalasi](#instalasi)
5. [Setup Docker & Environment](#setup-docker--environment)
6. [Week 1 — Ingestion Pipeline](#week-1--ingestion-pipeline)
7. [Week 2 — Risk Detection](#week-2--risk-detection)
8. [Week 3 — LLM Integration](#week-3--llm-integration)
9. [Week 4 — API Layer](#week-4--api-layer)
10. [Testing API](#testing-api)
11. [Troubleshooting](#troubleshooting)
12. [Arsitektur Lengkap](#arsitektur-lengkap)

---

## 🎯 Overview

**LegalDoc RAG** adalah sistem AI untuk analisis kontrak hukum menggunakan:
- **RAG (Retrieval-Augmented Generation)**: Kombinasi semantic search + LLM
- **Vector Database (Qdrant)**: Menyimpan embedding pasal-pasal kontrak
- **Legal-BERT**: Model embedding yang di-fine-tune untuk teks hukum
- **Llama 3 (via Groq)**: LLM untuk menjawab pertanyaan dan analisis

**Fitur:**
- 📄 Upload PDF kontrak → otomatis di-chunk dan di-index
- ❓ Q&A interaktif: "Apa risiko klausul indemnifikasi?"
- 📊 Ringkasan kontrak otomatis
- 🚨 Deteksi 10+ jenis klausul berbahaya (red flags)
- 📖 Penjelasan pasal dalam bahasa sederhana

---

## 📁 Struktur File Lengkap

```
legaldoc-rag/
│
├── ingestion/                  # Week 1 — Data Pipeline
│   ├── chunker.py              # Parsing PDF, split per pasal
│   ├── embedder.py             # Generate embedding dengan legal-BERT
│   ├── vector_store.py         # Simpan ke Qdrant
│   └── ocr.py                  # OCR untuk PDF scan (opsional)
│
├── risk_detector/              # Week 2 — Risk Detection
│   ├── detector.py             # Deteksi klausul berbahaya (cosine similarity)
│   └── templates/
│       └── __init__.py         # 10+ template klausul berbahaya
│
├── retrieval/                  # Week 3 — Retrieval
│   ├── searcher.py             # Search engine (query ke Qdrant)
│   └── reranker.py             # Re-ranking results (opsional)
│
├── llm/                        # Week 3 — LLM Integration
│   ├── prompts.py              # Template prompt untuk Q&A, summary, dll.
│   └── qa_chain.py             # RAG chain (retrieval + LLM)
│
├── api/                        # Week 4 — API Layer
│   ├── main.py                 # FastAPI app utama
│   └── routes/
│       ├── upload.py           # Endpoint upload PDF
│       ├── query.py            # Endpoint Q&A, summary, risks
│       └── __init__.py
│
├── config/
│   └── settings.py             # Konfigurasi dari .env
│
├── data/
│   ├── dummy/                  # PDF dummy untuk testing
│   ├── raw/                    # Upload directory
│   └── processed/              # OCR output (jika pakai OCR)
│
├── docker-compose.yml          # Setup Qdrant via Docker
├── requirements.txt            # Python dependencies
├── .env.example                # Template environment variables
└── README.md                   # Panduan ini
```

---

## ✅ Prasyarat

Pastikan ini sudah terinstall di komputer kamu:

| Software | Versi | Cara Cek |
|----------|-------|----------|
| Python | >= 3.11 | `python --version` |
| pip | terbaru | `pip --version` |
| Docker Desktop | terbaru | `docker --version` |
| Git | terbaru | `git --version` |

> **💡 Tips untuk Windows:** Gunakan **PowerShell** atau **Git Bash** untuk menjalankan perintah-perintah di panduan ini. Hindari Command Prompt (cmd) karena beberapa perintah mungkin berbeda.

---

## 🛠️ Instalasi

### Step 1 — Clone & Masuk ke Folder Project

```bash
# Kamu sudah di dalam folder ini, tapi pastikan kamu di root project
cd C:\Users\ASUS\Downloads\Jatis Mobile\stackplus-ai-legaldoc-rag-ai\legaldoc-rag
```

### Step 2 — Buat Virtual Environment

Virtual environment = "kotak terisolasi" untuk Python agar dependencies project ini
tidak bentrok dengan project lain di komputer kamu.

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
# venv\Scripts\activate.bat

# macOS / Linux:
# source venv/bin/activate
```

Setelah aktif, prompt terminal kamu akan berubah menjadi:
```
(venv) PS C:\...\legaldoc-rag>
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ **Proses ini akan memakan waktu 5-15 menit** karena:
> - `torch` berukuran ~2GB
> - `sentence-transformers` dan dependensinya juga besar
> 
> Pastikan koneksi internet stabil!

### Step 4 — Setup File `.env`

```bash
# Copy file contoh
copy .env.example .env   # Windows
# atau
cp .env.example .env     # macOS/Linux
```

Edit file `.env` dan isi sesuai kebutuhan:

```env
# Qdrant (sesuaikan jika port berbeda)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Model embedding (bisa diganti ke model lain)
EMBEDDING_MODEL=nlpaueb/legal-bert-base-uncased

# Groq API Key (isi nanti di Week 3)
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🐳 Setup Docker (Qdrant)

Qdrant (vector database) dijalankan via Docker. Pastikan Docker Desktop sudah berjalan.

### Cek isi docker-compose.yml

Pastikan file `docker-compose.yml` sudah ada dan isinya seperti ini (buat jika belum ada):

```yaml
version: "3.8"

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: legaldoc_qdrant
    ports:
      - "6333:6333"   # REST API port
      - "6334:6334"   # gRPC port
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_data:
```

### Jalankan Qdrant

```bash
# Jalankan Qdrant di background (detach mode)
docker-compose up -d qdrant

# Cek apakah container sudah jalan
docker ps

# Output yang diharapkan:
# CONTAINER ID   IMAGE             COMMAND      PORTS                    NAMES
# abc123...      qdrant/qdrant     "./qdrant"   0.0.0.0:6333->6333/tcp   legaldoc_qdrant
```

### Verifikasi Qdrant Berjalan

Buka browser dan akses: **http://localhost:6333/dashboard**

Kamu akan melihat Qdrant Web UI. Jika muncul → Qdrant sudah jalan! ✅

---

## 📦 Week 1 — Ingestion Pipeline

Ingestion pipeline terdiri dari 3 langkah:
**PDF → Chunk → Embed → Simpan ke Qdrant**

### Langkah 1 — Test Chunker

```bash
# Jalankan dari root folder project
python -m ingestion.chunker
```

**Output yang diharapkan:**
```
2024-01-01 10:00:00 [INFO] Membaca PDF: Kontrak_Sewa.pdf
2024-01-01 10:00:01 [INFO]   → 5 halaman, 8432 karakter diekstrak
2024-01-01 10:00:01 [INFO]   → 8 chunks berhasil dibuat dari 'Kontrak_Sewa.pdf'
...
============================================================
HASIL CHUNKING — 24 chunks total
============================================================
  [kontrak_sewa_pasal_1]
    Pasal   : 1
    Judul   : Definisi
    Sumber  : Kontrak_Sewa.pdf
    Preview : Pasal 1 - Definisi...
```

> **🔍 Apa yang terjadi?**
> `chunker.py` membaca setiap PDF, mencari pola "Pasal 1", "Pasal 2", dst.
> menggunakan regex, lalu memotong teks menjadi chunks berdasarkan pasal.

### Langkah 2 — Test Embedder

```bash
python -m ingestion.embedder
```

**Output yang diharapkan:**
```
Loading model: 'nlpaueb/legal-bert-base-uncased' (device=cpu)
  → Pertama kali download bisa makan waktu beberapa menit...
  ✓ Model loaded! Dimensi: 768, Waktu: 23.4s

[2] Embed single text...
    Output: array shape=(768,), dtype=float32

[3] Cosine Similarity Test...
    'pembayaran denda' vs 'biaya penalti' → similarity: 0.8234 (seharusnya tinggi)
    'pembayaran denda' vs 'kucing lucu'   → similarity: 0.1432 (seharusnya rendah)
    ✓ Test passed!
```

> ⏳ Download model legal-BERT pertama kali bisa makan 3-10 menit tergantung koneksi.
> Model akan di-cache, jadi percobaan berikutnya akan langsung.

### Langkah 3 — Jalankan Full Pipeline

Ini adalah langkah yang menjalankan ketiga proses sekaligus:

```bash
python -m ingestion.vector_store
```

**Output yang diharapkan:**
```
============================================================
MULAI INGESTION PIPELINE
============================================================

[Step 1/3] Chunking PDF...
  → 24 chunks dari 3 file PDF

[Step 2/3] Embedding chunks...
  ✓ Model loaded! Dimensi: 768
  Embedding 24 teks...
  ✓ Selesai! Shape: (24, 768)

[Step 3/3] Menyimpan ke Qdrant...
  ✓ Collection 'legaldoc_chunks' berhasil dibuat!
  → Batch 1: 24 points di-upsert
  ✓ Total 24 chunks berhasil disimpan!

============================================================
✓ INGESTION PIPELINE SELESAI!
  Chunks    : 24
  Upserted  : 24
  Files     : Kontrak_Kerja.pdf, Kontrak_Sewa.pdf, Kontrak_Vendor.pdf
============================================================
```

### Verifikasi Data Tersimpan di Qdrant

Buka: **http://localhost:6333/dashboard**

Di sidebar kiri, klik **Collections** → kamu akan melihat `legaldoc_chunks` dengan
jumlah vectors yang sesuai. ✅

---

## 🔍 Week 2 — Risk Detection

Risk detection menganalisis setiap pasal kontrak dan memberi label risiko.

### Langkah 1 — Lihat Template Klausul Berbahaya

```bash
python -m risk_detector.templates
```

**Output yang diharapkan:**
```
============================================================
RISK TEMPLATES — 10 klausul terdaftar
============================================================
  🔴 [RED] Indemnifikasi Sepihak
       5 contoh template
  🔴 [RED] Perpanjangan Otomatis (Auto-Renewal)
       5 contoh template
  🔴 [RED] Perubahan Sepihak (Unilateral Modification)
       5 contoh template
  ...
  🟡 [YELLOW] Denda Keterlambatan Tinggi
       5 contoh template
  ...
```

### Langkah 2 — Test Risk Detector

```bash
python -m risk_detector.detector
```

**Output yang diharapkan:**
```
Inisialisasi Risk Detector...
Menghitung centroid untuk 10 template risiko...
  ✓ Semua centroid selesai dihitung (10 kategori)

Menganalisis dokumen test...
  Pasal 1/4: Ketentuan Umum...
    🔴 RED FLAG: Perubahan Sepihak (sim=0.812)
  Pasal 2/4: Ganti Rugi...
    🔴 RED FLAG: Indemnifikasi Sepihak (sim=0.876)
  Pasal 3/4: Pembayaran dan Penyelesaian Sengketa...
    🟢 AMAN
  Pasal 4/4: Jangka Waktu...
    🟡 YELLOW: Perpanjangan Otomatis (sim=0.743)

======================================================================
LAPORAN RISIKO KONTRAK: test_contract.pdf
======================================================================
Total pasal dianalisis : 4
Red Flags              : 2
Yellow Flags           : 1
Aman                   : 1

⚠️  KONTRAK INI BERISIKO TINGGI! Ditemukan 2 klausul merah...
```

### Langkah 3 — Analisis PDF Kontrak Asli

Buat script sederhana untuk test dengan PDF dummy kamu:

```python
# Buat file: test_risk_pdf.py di root project
from risk_detector.detector import RiskDetector, print_risk_report

detector = RiskDetector()
report = detector.analyze_from_pdf("data/dummy/Kontrak_Sewa.pdf")
print_risk_report(report)
```

Jalankan:
```bash
python test_risk_pdf.py
```

---

## 🏗️ Penjelasan Arsitektur

```
PDF File
   │
   ▼
chunker.py ──── Baca teks dengan pypdf
   │             Deteksi "Pasal 1", "Pasal 2" dengan regex
   │             Split jadi Chunk objects dengan metadata
   │
   ▼
embedder.py ─── Load model legal-BERT dari HuggingFace
   │             Generate embedding (768 angka) per chunk
   │             Hasilkan list of {"id", "vector", "payload"}
   │
   ▼
vector_store.py ─ Konek ke Qdrant (Docker)
   │               Buat collection "legaldoc_chunks"
   │               Upsert semua vectors + metadata
   │
   │              ◄── WEEK 1 SELESAI ──────────────────────────────
   │
   ▼
detector.py ─── Load templates klausul berbahaya
   │             Embed semua template → hitung centroid per kategori
   │             Untuk setiap pasal:
   │               Embed pasal → hitung cosine similarity vs setiap centroid
   │               Sim > 0.72 → RED FLAG
   │               Sim > 0.55 → YELLOW FLAG
   │               Else → GREEN (aman)
   │             Buat ContractRiskReport
   │
   ▼
Risk Report ─── RED: Indemnifikasi (sim=0.87), Auto-Renewal (sim=0.76)
                YELLOW: Denda tinggi (sim=0.61)
                GREEN: 5 pasal aman
```

---

## 📅 Checklist Progress

### Week 1 ✅
- [ ] Setup virtual environment dan install dependencies
- [ ] Jalankan Docker Qdrant
- [ ] Test `chunker.py` — pastikan chunks ter-generate dengan benar
- [ ] Test `embedder.py` — pastikan model ter-load dan similarity test passed
- [ ] Jalankan full ingestion pipeline via `vector_store.py`
- [ ] Verifikasi data tersimpan di Qdrant dashboard

### Week 2 ✅
- [ ] Lihat semua template risiko via `templates/__init__.py`
- [ ] Test `detector.py` dengan dummy chunks
- [ ] Test `detector.py` dengan PDF kontrak asli
- [ ] Verifikasi Red Flag terdeteksi di pasal yang tepat
- [ ] Adjust threshold jika terlalu banyak/sedikit false positive

---

*Dokumentasi ini dibuat untuk LegalDoc RAG Project — Week 1 & 2*


## 🤖 Week 3 — LLM Integration

Week 3 menambahkan Llama 3 (via Groq API) untuk menjawab pertanyaan tentang kontrak.

### Setup Groq API Key

1. Daftar gratis di: **https://console.groq.com**
2. Buat API key baru
3. Copy API key (format: `gsk_...`)
4. Tambahkan ke `.env`:
   ```bash
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

### Langkah 1 — Test Prompt Templates

```bash
python -m llm.prompts
```

**Output yang diharapkan:**
```
[1] Q&A Prompt:
────────────────────────────────────────────────────────────────
Berdasarkan pasal-pasal kontrak berikut, jawab pertanyaan user.

CONTEXT (Pasal-pasal Relevan):
---
[Pasal 5 - Pembayaran]
...

[2] Summary Prompt:
────────────────────────────────────────────────────────────────
Buatkan ringkasan kontrak...
```

### Langkah 2 — Test Searcher (Retrieval)

```bash
python -m retrieval.searcher
```

**Output yang diharapkan:**
```
[1] Inisialisasi searcher...
  ✓ Searcher siap digunakan!

[2] Test search...
Query: 'Bagaimana cara pembayaran dalam kontrak ini?'
  → Ditemukan 2 chunks relevan
    1. [Kontrak_Sewa.pdf] Pasal 5 (score=0.872)
    2. [Kontrak_Sewa.pdf] Pasal 8 (score=0.645)
```

### Langkah 3 — Test RAG Chain (End-to-End)

```bash
python -m llm.qa_chain
```

**Output yang diharapkan:**
```
[1] Inisialisasi RAG chain...
  ✓ Groq client initialized (model=llama-3.3-70b-versatile)
  ✓ RAG Chain siap digunakan!

[2] Test Q&A...
────────────────────────────────────────────────────────────────
Q: Berapa denda jika terlambat bayar?
────────────────────────────────────────────────────────────────
A: Menurut Pasal 8 tentang Denda, keterlambatan pembayaran
   dikenakan denda sebesar 2% per bulan dari jumlah yang tertunggak.
   
   ⚠️ PERHATIAN: Denda 2% per bulan (24% per tahun) tergolong cukup
   tinggi. Sebaiknya negosiasikan grace period 7-14 hari sebelum
   denda mulai berlaku.

Sources:
  1. [Kontrak_Sewa.pdf] Pasal 8 (score=0.876)
  2. [Kontrak_Sewa.pdf] Pasal 5 (score=0.721)
```

> **🔍 Apa yang terjadi?**
> - Searcher mencari 5 pasal paling relevan dari Qdrant
> - Prompt builder menggabungkan pasal-pasal tersebut dengan pertanyaan
> - Groq LLM (Llama 3) menjawab berdasarkan pasal-pasal yang diberikan
> - LLM juga mendeteksi risiko dan memberikan warning

---

## 🌐 Week 4 — API Layer

Week 4 membungkus semua fungsi menjadi REST API dengan FastAPI.

### Langkah 1 — Setup Environment Variables

Pastikan `.env` sudah lengkap:

```bash
# Copy dari example jika belum
cp .env.example .env

# Edit dan isi GROQ_API_KEY
nano .env  # atau gunakan editor lain
```

Verifikasi settings:

```bash
python -m config.settings
```

**Output yang diharapkan:**
```
============================================================
SETTINGS VALIDATION
============================================================

Status: ✓ Valid

Settings Summary:
  qdrant              : localhost:6333
  model               : nlpaueb/legal-bert-base-uncased
  llm                 : llama-3.3-70b-versatile
  retrieval_top_k     : 5
```

### Langkah 2 — Jalankan API Server

```bash
# Dengan uvicorn (recommended)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Atau langsung via Python
python -m api.main
```

**Output yang diharapkan:**
```
============================================================
Starting LegalDoc RAG API v0.1.0
============================================================
✓ Qdrant connected: 24 vectors in collection 'legaldoc_chunks'
✓ Upload directory: data/raw
============================================================
API ready at http://0.0.0.0:8000
Docs at http://0.0.0.0:8000/docs
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Langkah 3 — Akses API Documentation

Buka browser: **http://localhost:8000/docs**

Kamu akan melihat **Swagger UI** dengan semua endpoint yang tersedia:

- 📤 **POST /upload/** — Upload PDF
- 📂 **GET /upload/files** — List file yang sudah di-upload
- ❓ **POST /query/ask** — Tanya pertanyaan
- 📊 **POST /query/summarize** — Ringkasan kontrak
- 📖 **POST /query/explain** — Jelaskan pasal tertentu
- 🚨 **POST /query/risks** — Deteksi risiko

---

## 🧪 Testing API

### Test 1 — Upload PDF

Di Swagger UI, expand **POST /upload/**, klik **Try it out**:

1. Click **Choose File** → pilih PDF dari `data/dummy/`
2. Set `process_immediately` = `true`
3. Click **Execute**

**Response:**
```json
{
  "success": true,
  "message": "File 'Kontrak_Sewa.pdf' berhasil di-upload dan di-ingest",
  "filename": "Kontrak_Sewa.pdf",
  "chunks_count": 8,
  "file_size_bytes": 125430
}
```

### Test 2 — Ask Question

Expand **POST /query/ask**, klik **Try it out**:

Request body:
```json
{
  "question": "Berapa denda jika terlambat bayar?",
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "Menurut Pasal 8 tentang Denda, keterlambatan pembayaran dikenakan denda sebesar 2% per bulan dari jumlah yang tertunggak...",
  "sources": [
    {
      "chunk_id": "kontrak_sewa_pasal_8",
      "pasal_number": 8,
      "pasal_title": "Denda",
      "source_file": "Kontrak_Sewa.pdf",
      "relevance_score": 0.876
    }
  ]
}
```

### Test 3 — Detect Risks

Expand **POST /query/risks**, klik **Try it out**:

Request body:
```json
{
  "filter_source": "Kontrak_Sewa.pdf",
  "use_llm": false
}
```

**Response:**
```json
{
  "overall_risk_level": "YELLOW",
  "risk_summary": "Kontrak perlu perhatian. Ditemukan 2 klausul yang perlu dicek.",
  "red_flags": [],
  "yellow_flags": [
    {
      "name": "Denda Keterlambatan Tinggi",
      "risk_level": "YELLOW",
      "pasal_number": 8,
      "pasal_title": "Denda",
      "description": "Denda 2% per bulan tergolong tinggi...",
      "advice": "Negosiasikan ke bawah atau tambahkan grace period 7-14 hari",
      "similarity_score": 0.68
    }
  ]
}
```

### Test dengan cURL (Command Line)

```bash
# Test upload
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@data/dummy/Kontrak_Sewa.pdf" \
  -F "process_immediately=true"

# Test ask
curl -X POST "http://localhost:8000/query/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa risiko klausul indemnifikasi?",
    "include_sources": true
  }'

# Test summarize
curl -X POST "http://localhost:8000/query/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "focus_areas": ["pembayaran", "terminasi", "risiko"]
  }'
```

### Test dengan Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Upload PDF
with open("data/dummy/Kontrak_Sewa.pdf", "rb") as f:
    files = {"file": f}
    data = {"process_immediately": True}
    response = requests.post(f"{BASE_URL}/upload/", files=files, data=data)
    print(response.json())

# Ask question
payload = {
    "question": "Bagaimana cara pembayaran?",
    "include_sources": True
}
response = requests.post(f"{BASE_URL}/query/ask", json=payload)
print(response.json())

# Summarize
payload = {"focus_areas": ["pembayaran", "denda"]}
response = requests.post(f"{BASE_URL}/query/summarize", json=payload)
print(response.json()["summary"])
```

---

## 🏗️ Arsitektur Lengkap

```
┌─────────────┐
│   User      │
│  (Frontend) │
└──────┬──────┘
       │ HTTP Request
       ▼
┌────────────────────────────────────────────────┐
│          FastAPI (api/main.py)                 │
│  ┌────────────────────┬──────────────────────┐ │
│  │  /upload/          │  /query/             │ │
│  │  - POST /          │  - POST /ask         │ │
│  │  - GET /files      │  - POST /summarize   │ │
│  │  - DELETE /files/  │  - POST /explain     │ │
│  │                    │  - POST /risks       │ │
│  └────────┬───────────┴──────────┬───────────┘ │
└───────────┼──────────────────────┼─────────────┘
            │                      │
            ▼                      ▼
    ┌──────────────┐      ┌────────────────┐
    │  Ingestion   │      │   RAG Chain    │
    │  Pipeline    │      │  (qa_chain.py) │
    └───────┬──────┘      └────┬───────────┘
            │                  │
            │                  ├──► Searcher ──┐
            │                  │               │
            │                  └──► LLM (Groq) │
            │                                  │
            ▼                                  ▼
    ┌──────────────────────────────────────────────┐
    │          Qdrant Vector Database              │
    │      (Docker container di port 6333)         │
    │                                              │
    │  Collection: legaldoc_chunks                 │
    │  - chunk_id, vector (768 dim), metadata      │
    └──────────────────────────────────────────────┘
            ▲
            │ Embedding
            │
    ┌───────────────┐
    │  Legal-BERT   │
    │  (embedder)   │
    └───────────────┘
```

**Alur Request `/query/ask`:**
1. User kirim pertanyaan via POST /query/ask
2. API route `query.py` terima request
3. `qa_chain.ask()` dipanggil
4. **Retrieval**: Searcher embed query → cari di Qdrant → dapat 5 chunks relevan
5. **Prompt**: Build prompt dengan chunks + question
6. **Generation**: Kirim prompt ke Groq (Llama 3) → dapat jawaban
7. Return jawaban + sources (opsional) ke user

---

## ❗ Troubleshooting

### Error: `GROQ_API_KEY belum di-set`

**Solusi:**
1. Daftar di https://console.groq.com
2. Buat API key
3. Tambahkan ke `.env`: `GROQ_API_KEY=gsk_...`
4. Restart API server

---

### Error: `Connection refused` saat akses API

**Penyebab:** Server belum jalan atau salah port.

**Solusi:**
```bash
# Cek apakah server jalan
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kalau belum jalan, start:
uvicorn api.main:app --reload

# Akses di browser:
http://localhost:8000/docs
```

---

### Error: `405 Method Not Allowed`

**Penyebab:** Salah method (GET vs POST).

**Solusi:** Cek dokumentasi di /docs, pastikan pakai method yang benar:
- `/upload/` → POST (bukan GET)
- `/query/ask` → POST (bukan GET)

---

### Upload file lambat / timeout

**Penyebab:** PDF besar dan embedding lambat (pakai CPU).

**Solusi:**
- Gunakan `process_immediately=false` → proses di background
- Atau set timeout lebih tinggi di client
- Untuk production: gunakan GPU atau model embedding lebih kecil

---

### LLM response ngaco / halusinasi

**Penyebab:**
1. Query terlalu ambigu
2. Tidak ada chunks relevan (threshold terlalu tinggi)
3. Context terlalu sedikit

**Solusi:**
- Cek `sources` di response → apakah chunks yang di-retrieve relevan?
- Turunkan `RETRIEVAL_SCORE_THRESHOLD` di `.env` (misal: 0.3)
- Naikkan `RETRIEVAL_TOP_K` (misal: 10)
- Perbaiki prompt di `llm/prompts.py`

---

## 📅 Checklist Progress Week 1-4

### Week 1 ✅ — Ingestion
- [ ] Setup virtual environment dan install dependencies
- [ ] Jalankan Docker Qdrant
- [ ] Test `chunker.py` — chunks ter-generate dengan benar
- [ ] Test `embedder.py` — model ter-load dan similarity test passed
- [ ] Jalankan full ingestion pipeline
- [ ] Verifikasi data tersimpan di Qdrant dashboard

### Week 2 ✅ — Risk Detection
- [ ] Lihat semua template risiko
- [ ] Test `detector.py` dengan dummy chunks
- [ ] Test `detector.py` dengan PDF kontrak asli
- [ ] Verifikasi Red Flag terdeteksi di pasal yang tepat

### Week 3 ✅ — LLM Integration
- [ ] Daftar Groq API dan dapatkan API key
- [ ] Set `GROQ_API_KEY` di `.env`
- [ ] Test `prompts.py` — prompt templates OK
- [ ] Test `searcher.py` — retrieval berfungsi
- [ ] Test `qa_chain.py` — RAG end-to-end berfungsi

### Week 4 ✅ — API Layer
- [ ] Validasi settings dengan `config.settings`
- [ ] Jalankan API server
- [ ] Akses Swagger UI di /docs
- [ ] Test upload PDF via API
- [ ] Test ask question via API
- [ ] Test summarize via API
- [ ] Test detect risks via API
- [ ] Test dengan cURL atau Python client

---

## 🚀 Next Steps (Week 5-6)

- [ ] **Testing & Evaluation**
  - Buat 50 PDF kontrak dummy dengan variasi
  - Hitung RAGAS score (faithfulness, relevancy)
  - A/B testing prompt variations
  
- [ ] **Deployment**
  - Dockerize seluruh aplikasi (API + Qdrant + dependencies)
  - Deploy ke cloud (Railway, Render, atau DigitalOcean)
  - Setup monitoring (logging, error tracking)
  
- [ ] **Frontend** (Opsional)
  - Buat UI dengan React/Next.js
  - Upload PDF, chat interface
  - Visualisasi risk report

- [ ] **Optimization**
  - Re-ranker untuk meningkatkan akurasi retrieval
  - Caching untuk response yang sering ditanya
  - Batch processing untuk banyak PDF

---

*Dokumentasi ini dibuat untuk LegalDoc RAG Project — Complete Guide Week 1-4*
*Jika ada pertanyaan atau issue, silakan buat issue di repository atau hubungi tim.*