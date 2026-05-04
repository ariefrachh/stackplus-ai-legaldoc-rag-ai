"""
detector.py — Deteksi klausul berbahaya menggunakan cosine similarity

Cara kerja:
  1. Load semua template klausul berbahaya dari templates/risk_templates.py
  2. Embed setiap template teks menggunakan legal-BERT
  3. Hitung "centroid" embedding untuk setiap kategori risiko
     (centroid = rata-rata dari semua embedding template dalam satu kategori)
  4. Untuk setiap pasal kontrak:
     a. Embed pasal tersebut
     b. Hitung cosine similarity dengan setiap centroid
     c. Jika similarity > RED_THRESHOLD → label "RED"
        Jika similarity > YELLOW_THRESHOLD → label "YELLOW"
        Jika tidak ada yang terdeteksi → label "GREEN"

Mengapa rata-rata embedding (centroid)?
  Bayangkan kamu punya 5 cara untuk mengatakan "denda keterlambatan".
  Jika kamu rata-rata semua embedding-nya, kamu dapat satu titik di tengah
  yang paling "representatif" untuk konsep denda keterlambatan.
  Ini lebih robust dibanding hanya pakai satu contoh.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from risk_detector.templates import RISK_TEMPLATES, RiskTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Threshold Similarity ────────────────────────────────────────────────────────
#
# Nilai ini menentukan seberapa "mirip" sebuah pasal harus dengan template
# agar di-flag sebagai berbahaya.
#
# Range: 0.0 (tidak mirip sama sekali) sampai 1.0 (identik)
# Tips: Jika terlalu banyak false positive, naikkan threshold-nya.
#       Jika terlalu banyak yang missed, turunkan threshold-nya.
#
RED_THRESHOLD = 0.72     # Pasal sangat mirip template berbahaya → Red Flag
YELLOW_THRESHOLD = 0.55  # Pasal cukup mirip → Yellow Flag (perlu dicek manual)


# ─── Data Model ──────────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    """
    Hasil analisis risiko untuk satu pasal kontrak.
    
    Attributes:
        chunk_id     : ID pasal yang dianalisis
        pasal_number : Nomor pasal
        pasal_title  : Judul pasal
        text_preview : 200 karakter pertama teks pasal
        risk_level   : "GREEN", "YELLOW", atau "RED"
        detected_risks: List of risiko yang terdeteksi
        max_similarity: Similarity score tertinggi yang ditemukan
    """
    chunk_id: str
    pasal_number: int
    pasal_title: str
    text_preview: str
    risk_level: str  # "GREEN" | "YELLOW" | "RED"
    detected_risks: list[dict] = field(default_factory=list)
    max_similarity: float = 0.0
    
    @property
    def icon(self) -> str:
        return {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(self.risk_level, "⚪")
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "pasal_number": self.pasal_number,
            "pasal_title": self.pasal_title,
            "text_preview": self.text_preview,
            "risk_level": self.risk_level,
            "detected_risks": self.detected_risks,
            "max_similarity": round(self.max_similarity, 4),
        }


@dataclass
class ContractRiskReport:
    """
    Laporan risiko untuk keseluruhan kontrak.
    
    Attributes:
        source_file   : Nama file kontrak
        total_clauses : Total pasal yang dianalisis
        red_flags     : Pasal-pasal dengan Red Flag
        yellow_flags  : Pasal-pasal dengan Yellow Flag
        green_clauses : Pasal-pasal aman
        risk_summary  : Ringkasan risiko dalam bahasa sederhana
    """
    source_file: str
    total_clauses: int
    red_flags: list[RiskResult]
    yellow_flags: list[RiskResult]
    green_clauses: list[RiskResult]
    
    @property
    def overall_risk_level(self) -> str:
        """Tingkat risiko keseluruhan kontrak."""
        if self.red_flags:
            return "RED"
        elif self.yellow_flags:
            return "YELLOW"
        return "GREEN"
    
    @property
    def risk_summary(self) -> str:
        """Ringkasan risiko dalam bahasa sederhana."""
        if self.overall_risk_level == "RED":
            names = [r.detected_risks[0]["name"] for r in self.red_flags if r.detected_risks]
            return (
                f"⚠️  KONTRAK INI BERISIKO TINGGI! "
                f"Ditemukan {len(self.red_flags)} klausul merah: {', '.join(names[:3])}."
            )
        elif self.overall_risk_level == "YELLOW":
            return (
                f"⚡ Kontrak perlu perhatian. "
                f"Ditemukan {len(self.yellow_flags)} klausul yang perlu dicek."
            )
        return "✅ Kontrak tampak aman. Tidak ditemukan klausul berbahaya yang signifikan."
    
    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "total_clauses": self.total_clauses,
            "overall_risk_level": self.overall_risk_level,
            "risk_summary": self.risk_summary,
            "red_count": len(self.red_flags),
            "yellow_count": len(self.yellow_flags),
            "green_count": len(self.green_clauses),
            "red_flags": [r.to_dict() for r in self.red_flags],
            "yellow_flags": [r.to_dict() for r in self.yellow_flags],
        }


# ─── RiskDetector Class ──────────────────────────────────────────────────────────

class RiskDetector:
    """
    Deteksi klausul berbahaya dalam kontrak menggunakan cosine similarity.
    
    Workflow:
    1. risk_templates: Load embedder dan compute centroid untuk setiap template
    2. analyze_chunk: Analisis satu pasal
    3. analyze_document: Analisis seluruh kontrak
    
    Usage:
        detector = RiskDetector()
        report = detector.analyze_document(chunks, source_file="Kontrak_Sewa.pdf")
        print(report.risk_summary)
    """
    
    def __init__(
        self,
        embedder=None,  # LegalEmbedder instance (opsional, akan dibuat jika None)
        templates: Optional[list[RiskTemplate]] = None,
        red_threshold: float = RED_THRESHOLD,
        yellow_threshold: float = YELLOW_THRESHOLD,
    ):
        """
        Inisialisasi detector dan pre-compute centroid embedding untuk semua template.
        
        Args:
            embedder        : LegalEmbedder instance. Jika None, akan dibuat baru.
            templates       : List of RiskTemplate. Jika None, pakai default dari templates/__init__.py
            red_threshold   : Threshold cosine similarity untuk Red Flag
            yellow_threshold: Threshold cosine similarity untuk Yellow Flag
        """
        self.red_threshold = red_threshold
        self.yellow_threshold = yellow_threshold
        self.templates = templates or RISK_TEMPLATES
        
        # Load embedder
        if embedder is None:
            logger.info("Inisialisasi embedder untuk risk detection...")
            from ingestion.embedder import LegalEmbedder
            self.embedder = LegalEmbedder()
        else:
            self.embedder = embedder
        
        # Pre-compute centroid embedding untuk setiap template kategori
        # Ini dilakukan sekali saat init, bukan setiap kali detect
        self.template_centroids: dict[str, dict] = {}
        self._compute_template_centroids()
    
    def _compute_template_centroids(self) -> None:
        """
        Hitung centroid embedding untuk setiap kategori template.
        
        Centroid = rata-rata dari semua embedding template dalam satu kategori.
        
        Contoh:
        Template "Indemnifikasi Sepihak" punya 5 contoh teks.
        → Embed semua 5 teks → dapat 5 vector (768,)
        → Rata-rata 5 vector → dapat 1 centroid vector (768,)
        → Centroid ini merepresentasikan "konsep indemnifikasi berbahaya"
        """
        logger.info(f"Menghitung centroid untuk {len(self.templates)} template risiko...")
        
        for template in self.templates:
            # Embed semua contoh teks dalam template ini
            embeddings = self.embedder.embed_batch(
                template.templates,
                batch_size=len(template.templates),
            )
            
            # Hitung rata-rata (centroid)
            # axis=0 berarti rata-rata per dimensi (bukan per vector)
            centroid = np.mean(embeddings, axis=0)
            
            # Normalisasi centroid agar bisa pakai dot product sebagai cosine similarity
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            
            self.template_centroids[template.name] = {
                "centroid": centroid,
                "template": template,
            }
            
            logger.debug(f"  ✓ Centroid '{template.name}': shape={centroid.shape}")
        
        logger.info(f"  ✓ Semua centroid selesai dihitung ({len(self.template_centroids)} kategori)")
    
    def analyze_chunk(self, chunk) -> RiskResult:
        """
        Analisis satu pasal kontrak untuk mendeteksi risiko.
        
        Proses:
        1. Embed teks pasal
        2. Hitung cosine similarity dengan setiap template centroid
        3. Flag berdasarkan threshold
        
        Args:
            chunk: Chunk object dari chunker.py
            
        Returns:
            RiskResult dengan informasi risiko yang terdeteksi
        """
        # Embed teks pasal (gabungkan judul + isi untuk konteks yang lebih baik)
        text_to_embed = f"{chunk.pasal_title}\n\n{chunk.text}"
        chunk_vector = self.embedder.embed_text(text_to_embed)
        
        # Hitung similarity dengan setiap template centroid
        detected_risks = []
        
        for name, data in self.template_centroids.items():
            centroid = data["centroid"]
            template = data["template"]
            
            # Cosine similarity = dot product (karena keduanya sudah dinormalisasi)
            similarity = float(np.dot(chunk_vector, centroid))
            
            # Tentukan apakah ini red atau yellow berdasarkan threshold
            # dan risk_level template itu sendiri
            if template.risk_level == "RED" and similarity >= self.red_threshold:
                detected_risks.append({
                    "name": name,
                    "risk_level": "RED",
                    "similarity": round(similarity, 4),
                    "description": template.description,
                    "advice": template.advice,
                    "tags": template.tags,
                })
            elif template.risk_level in ("RED", "YELLOW") and similarity >= self.yellow_threshold:
                # Jika template RED tapi similarity-nya di antara yellow dan red threshold
                # → tetap jadi YELLOW warning
                effective_level = "RED" if (
                    template.risk_level == "RED" and similarity >= self.red_threshold
                ) else "YELLOW"
                
                detected_risks.append({
                    "name": name,
                    "risk_level": effective_level,
                    "similarity": round(similarity, 4),
                    "description": template.description,
                    "advice": template.advice,
                    "tags": template.tags,
                })
        
        # Urutkan dari similarity tertinggi
        detected_risks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Tentukan risk_level keseluruhan pasal ini
        if any(r["risk_level"] == "RED" for r in detected_risks):
            overall_level = "RED"
        elif detected_risks:
            overall_level = "YELLOW"
        else:
            overall_level = "GREEN"
        
        max_sim = detected_risks[0]["similarity"] if detected_risks else 0.0
        
        return RiskResult(
            chunk_id=chunk.chunk_id,
            pasal_number=chunk.pasal_number,
            pasal_title=chunk.pasal_title,
            text_preview=chunk.text[:200].replace("\n", " "),
            risk_level=overall_level,
            detected_risks=detected_risks,
            max_similarity=max_sim,
        )
    
    def analyze_document(
        self,
        chunks: list,  # list[Chunk]
        source_file: str = "unknown",
    ) -> ContractRiskReport:
        """
        Analisis seluruh kontrak (semua pasal) dan buat laporan risiko.
        
        Args:
            chunks     : List of Chunk dari chunker.py
            source_file: Nama file kontrak (untuk laporan)
            
        Returns:
            ContractRiskReport dengan ringkasan dan detail semua risiko
        """
        logger.info(f"Menganalisis risiko untuk '{source_file}' ({len(chunks)} pasal)...")
        
        red_flags = []
        yellow_flags = []
        green_clauses = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"  Pasal {i}/{len(chunks)}: {chunk.pasal_title[:40]}...")
            result = self.analyze_chunk(chunk)
            
            if result.risk_level == "RED":
                red_flags.append(result)
                logger.warning(
                    f"    🔴 RED FLAG: {result.detected_risks[0]['name']} "
                    f"(sim={result.max_similarity:.3f})"
                )
            elif result.risk_level == "YELLOW":
                yellow_flags.append(result)
                logger.info(
                    f"    🟡 YELLOW: {result.detected_risks[0]['name']} "
                    f"(sim={result.max_similarity:.3f})"
                )
            else:
                green_clauses.append(result)
                logger.debug(f"    🟢 AMAN")
        
        report = ContractRiskReport(
            source_file=source_file,
            total_clauses=len(chunks),
            red_flags=red_flags,
            yellow_flags=yellow_flags,
            green_clauses=green_clauses,
        )
        
        logger.info(f"  ✓ Analisis selesai: {len(red_flags)} RED, {len(yellow_flags)} YELLOW, {len(green_clauses)} GREEN")
        logger.info(f"  → {report.risk_summary}")
        
        return report
    
    def analyze_from_pdf(self, pdf_path: str) -> ContractRiskReport:
        """
        Shortcut: Analisis langsung dari file PDF (chunk + detect sekaligus).
        
        Args:
            pdf_path: Path ke file PDF kontrak
            
        Returns:
            ContractRiskReport
        """
        from pathlib import Path
        from ingestion.chunker import chunk_pdf
        
        path = Path(pdf_path)
        chunks = chunk_pdf(path)
        return self.analyze_document(chunks, source_file=path.name)


# ─── Fungsi Utility untuk Print Report ──────────────────────────────────────────

def print_risk_report(report: ContractRiskReport) -> None:
    """Cetak laporan risiko ke terminal dalam format yang mudah dibaca."""
    print("\n" + "="*70)
    print(f"LAPORAN RISIKO KONTRAK: {report.source_file}")
    print("="*70)
    print(f"Total pasal dianalisis : {report.total_clauses}")
    print(f"Red Flags              : {len(report.red_flags)}")
    print(f"Yellow Flags           : {len(report.yellow_flags)}")
    print(f"Aman                   : {len(report.green_clauses)}")
    print(f"\n{report.risk_summary}")
    
    if report.red_flags:
        print(f"\n{'─'*70}")
        print("🔴 RED FLAGS — Klausul Berbahaya (Segera Tindaklanjuti!)")
        print(f"{'─'*70}")
        for result in report.red_flags:
            print(f"\n  Pasal {result.pasal_number}: {result.pasal_title}")
            print(f"  Preview: {result.text_preview[:100]}...")
            for risk in result.detected_risks:
                if risk["risk_level"] == "RED":
                    print(f"\n  ⚠️  Risiko: {risk['name']} (similarity: {risk['similarity']:.3f})")
                    print(f"     Kenapa berbahaya: {risk['description'][:150]}...")
                    print(f"     Saran: {risk['advice'][:150]}...")
    
    if report.yellow_flags:
        print(f"\n{'─'*70}")
        print("🟡 YELLOW FLAGS — Perlu Perhatian (Cek Sebelum Tanda Tangan!)")
        print(f"{'─'*70}")
        for result in report.yellow_flags:
            print(f"\n  Pasal {result.pasal_number}: {result.pasal_title}")
            if result.detected_risks:
                risk = result.detected_risks[0]
                print(f"  ⚡ {risk['name']} (similarity: {risk['similarity']:.3f})")
    
    print(f"\n{'='*70}\n")


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Jalankan dengan: python -m risk_detector.detector
    
    Test dengan teks kontrak dummy — tidak perlu PDF.
    """
    from ingestion.chunker import Chunk
    
    print("="*60)
    print("TEST RISK DETECTOR")
    print("="*60)
    
    # Buat dummy chunks untuk testing
    test_chunks = [
        Chunk(
            chunk_id="test_pasal_1",
            text=(
                "Pihak pertama berhak mengubah, menambah, atau mengurangi ketentuan "
                "perjanjian ini sewaktu-waktu tanpa persetujuan pihak kedua."
            ),
            pasal_number=1,
            pasal_title="Ketentuan Umum",
            source_file="test_contract.pdf",
        ),
        Chunk(
            chunk_id="test_pasal_2",
            text=(
                "Pihak kedua wajib membebaskan dan mengganti rugi pihak pertama dari "
                "segala klaim, kerugian, denda, dan biaya apapun yang timbul."
            ),
            pasal_number=2,
            pasal_title="Ganti Rugi",
            source_file="test_contract.pdf",
        ),
        Chunk(
            chunk_id="test_pasal_3",
            text=(
                "Pembayaran dilakukan setiap tanggal 1 bulan berjalan. "
                "Kedua pihak sepakat untuk menyelesaikan sengketa secara musyawarah mufakat."
            ),
            pasal_number=3,
            pasal_title="Pembayaran dan Penyelesaian Sengketa",
            source_file="test_contract.pdf",
        ),
        Chunk(
            chunk_id="test_pasal_4",
            text=(
                "Perjanjian ini akan diperpanjang secara otomatis selama 1 tahun "
                "kecuali ada pemberitahuan penghentian 90 hari sebelum berakhir."
            ),
            pasal_number=4,
            pasal_title="Jangka Waktu",
            source_file="test_contract.pdf",
        ),
    ]
    
    print("\nInisialisasi Risk Detector...")
    detector = RiskDetector()
    
    print("\nMenganalisis dokumen test...")
    report = detector.analyze_document(test_chunks, source_file="test_contract.pdf")
    
    print_risk_report(report)
    
    # Verifikasi hasil
    print("VERIFIKASI:")
    print(f"  Red flags ditemukan   : {len(report.red_flags)} (diharapkan >= 1)")
    print(f"  Yellow flags ditemukan: {len(report.yellow_flags)}")
    print(f"  Pasal aman            : {len(report.green_clauses)}")