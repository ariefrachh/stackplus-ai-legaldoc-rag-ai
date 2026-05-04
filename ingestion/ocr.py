"""
ocr.py — OCR untuk PDF hasil scan (gambar, bukan teks digital)

Kapan pakai OCR?
  Tidak semua PDF bisa langsung diekstrak teksnya.
  PDF ada dua jenis:
  1. "Digital PDF" → dibuat dari Word/Google Docs, bisa langsung ekstrak teks (pakai pypdf)
  2. "Scanned PDF" → hasil scan dokumen fisik, isinya gambar → perlu OCR

  OCR (Optical Character Recognition) = teknologi untuk "membaca" teks dari gambar.

Pipeline OCR di sini:
  PDF → Konversi setiap halaman jadi gambar (PNG) → OCR setiap gambar → Gabungkan teks

Library yang dipakai:
  - pdf2image: Konversi halaman PDF ke gambar
  - pytesseract: Python wrapper untuk Tesseract OCR engine
  
Instalasi dependensi sistem (wajib):
  - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
    Windows: choco install tesseract
    macOS  : brew install tesseract
    Linux  : sudo apt install tesseract-ocr tesseract-ocr-ind

  - Poppler (untuk pdf2image):
    Windows: https://github.com/oschwartz10612/poppler-windows/releases/
    macOS  : brew install poppler
    Linux  : sudo apt install poppler-utils
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _check_dependencies() -> dict[str, bool]:
    """
    Cek apakah semua dependensi OCR sudah terinstall.
    
    Returns:
        Dict {"pytesseract": bool, "pdf2image": bool, "tesseract_binary": bool}
    """
    status = {}
    
    # Cek pytesseract Python library
    try:
        import pytesseract
        status["pytesseract"] = True
    except ImportError:
        status["pytesseract"] = False
    
    # Cek pdf2image Python library
    try:
        import pdf2image
        status["pdf2image"] = True
    except ImportError:
        status["pdf2image"] = False
    
    # Cek Tesseract binary di sistem
    status["tesseract_binary"] = shutil.which("tesseract") is not None
    
    return status


def is_ocr_available() -> bool:
    """Return True jika semua dependensi OCR tersedia."""
    deps = _check_dependencies()
    return all(deps.values())


def is_scanned_pdf(pdf_path: Path, text_threshold: int = 50) -> bool:
    """
    Deteksi apakah sebuah PDF adalah hasil scan (gambar) atau digital PDF.
    
    Cara deteksi:
    Coba ekstrak teks dari halaman pertama.
    Jika teks yang diekstrak < threshold karakter → kemungkinan besar PDF scan.
    
    Args:
        pdf_path      : Path ke file PDF
        text_threshold: Minimum jumlah karakter untuk dianggap "digital PDF"
        
    Returns:
        True jika PDF tampaknya hasil scan, False jika digital PDF
    """
    try:
        import pypdf
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            if not reader.pages:
                return True
            first_page_text = reader.pages[0].extract_text() or ""
            return len(first_page_text.strip()) < text_threshold
    except Exception:
        return True  # Jika gagal baca, anggap scan


def ocr_pdf(
    pdf_path: str | Path,
    output_text_path: Optional[str | Path] = None,
    language: str = "ind+eng",  # Bahasa Indonesia + Inggris
    dpi: int = 300,             # Resolusi gambar untuk OCR (makin tinggi = lebih akurat, lebih lambat)
) -> str:
    """
    Jalankan OCR pada PDF scan dan kembalikan teks hasilnya.
    
    Args:
        pdf_path        : Path ke file PDF scan
        output_text_path: Jika diberikan, simpan hasil OCR ke file .txt ini
        language        : Kode bahasa Tesseract (ind=Indonesia, eng=English)
        dpi             : Resolusi konversi PDF ke gambar (300 DPI direkomendasikan)
        
    Returns:
        String teks hasil OCR dari seluruh halaman
        
    Raises:
        ImportError    : Jika library OCR tidak terinstall
        RuntimeError   : Jika Tesseract binary tidak ditemukan
        FileNotFoundError: Jika file PDF tidak ada
    """
    pdf_path = Path(pdf_path)
    
    # Cek file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")
    
    # Cek dependensi
    deps = _check_dependencies()
    if not deps["pytesseract"]:
        raise ImportError(
            "pytesseract tidak terinstall. Jalankan: pip install pytesseract"
        )
    if not deps["pdf2image"]:
        raise ImportError(
            "pdf2image tidak terinstall. Jalankan: pip install pdf2image"
        )
    if not deps["tesseract_binary"]:
        raise RuntimeError(
            "Tesseract OCR binary tidak ditemukan di sistem.\n"
            "Windows: choco install tesseract\n"
            "macOS  : brew install tesseract tesseract-lang\n"
            "Linux  : sudo apt install tesseract-ocr tesseract-ocr-ind"
        )
    
    import pytesseract
    from pdf2image import convert_from_path
    
    logger.info(f"Memulai OCR pada: {pdf_path.name} (DPI={dpi}, lang={language})")
    
    # Step 1: Konversi setiap halaman PDF menjadi gambar
    logger.info("  → Konversi PDF ke gambar...")
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        raise RuntimeError(
            f"Gagal konversi PDF ke gambar.\n"
            f"Pastikan Poppler sudah terinstall.\n"
            f"Detail: {e}"
        ) from e
    
    logger.info(f"  → {len(images)} halaman ditemukan, mulai OCR...")
    
    # Step 2: Jalankan OCR pada setiap gambar
    all_pages_text = []
    for i, image in enumerate(images, start=1):
        logger.info(f"  → OCR halaman {i}/{len(images)}...")
        
        # Tesseract config: --oem 3 = LSTM OCR Engine, --psm 3 = Auto page segmentation
        custom_config = r"--oem 3 --psm 3"
        
        page_text = pytesseract.image_to_string(
            image,
            lang=language,
            config=custom_config,
        )
        all_pages_text.append(f"--- Halaman {i} ---\n{page_text}")
    
    # Step 3: Gabungkan semua teks
    full_text = "\n\n".join(all_pages_text)
    
    logger.info(f"  ✓ OCR selesai! Total {len(full_text)} karakter diekstrak")
    
    # Step 4: Simpan ke file jika diminta
    if output_text_path:
        output_text_path = Path(output_text_path)
        output_text_path.parent.mkdir(parents=True, exist_ok=True)
        output_text_path.write_text(full_text, encoding="utf-8")
        logger.info(f"  → Hasil OCR disimpan ke: {output_text_path}")
    
    return full_text


def ocr_pdf_if_needed(
    pdf_path: str | Path,
    processed_dir: str | Path = "data/processed",
) -> Path:
    """
    Smart OCR: Cek dulu apakah PDF butuh OCR, jika ya jalankan OCR, jika tidak skip.
    Hasil OCR disimpan sebagai file .txt di processed_dir.
    
    Fungsi ini dipanggil oleh pipeline ingestion sebelum chunking.
    Alur: PDF → (cek scan?) → [OCR jika scan] → file .txt → chunker.py
    
    Args:
        pdf_path     : Path ke file PDF
        processed_dir: Direktori untuk menyimpan hasil OCR
        
    Returns:
        Path ke file .txt hasil OCR (atau path PDF asli jika tidak perlu OCR)
    """
    pdf_path = Path(pdf_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Cek apakah sudah pernah di-OCR sebelumnya (cache)
    output_txt = processed_dir / (pdf_path.stem + "_ocr.txt")
    if output_txt.exists():
        logger.info(f"OCR cache ditemukan: {output_txt.name} — skip re-OCR")
        return output_txt
    
    # Deteksi apakah perlu OCR
    if is_scanned_pdf(pdf_path):
        logger.info(f"'{pdf_path.name}' terdeteksi sebagai PDF scan → menjalankan OCR...")
        
        if not is_ocr_available():
            logger.warning(
                "OCR diperlukan tapi library/binary tidak tersedia. "
                "PDF ini akan diproses tanpa OCR (kemungkinan hasilnya buruk)."
            )
            return pdf_path
        
        ocr_pdf(pdf_path, output_text_path=output_txt)
        return output_txt
    else:
        logger.info(f"'{pdf_path.name}' adalah digital PDF — OCR tidak diperlukan")
        return pdf_path


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Jalankan dengan: python -m ingestion.ocr
    Akan cek dependensi OCR dan test deteksi scan PDF.
    """
    print("="*60)
    print("CEK DEPENDENSI OCR")
    print("="*60)
    
    deps = _check_dependencies()
    for dep, available in deps.items():
        status = "✓ Tersedia" if available else "✗ Tidak tersedia"
        print(f"  {dep:25s}: {status}")
    
    print()
    if is_ocr_available():
        print("✓ Semua dependensi OCR siap digunakan!")
    else:
        print("✗ Beberapa dependensi OCR belum tersedia.")
        print("  Jalankan perintah instalasi di atas, lalu coba lagi.")
    
    # Test deteksi scan PDF jika ada sample
    from pathlib import Path
    sample = Path("data/dummy/Kontrak_Sewa.pdf")
    if sample.exists():
        print(f"\nTest deteksi PDF: {sample.name}")
        is_scan = is_scanned_pdf(sample)
        print(f"  → Terdeteksi sebagai: {'PDF Scan (butuh OCR)' if is_scan else 'Digital PDF (tidak butuh OCR)'}")