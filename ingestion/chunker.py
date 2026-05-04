"""
chunker.py — Parsing PDF dan split per pasal dengan metadata

Cara kerja:
1. Baca file PDF menggunakan pypdf
2. Gabungkan semua teks dari setiap halaman
3. Gunakan regex untuk mendeteksi pola nomor pasal (Pasal 1, PASAL I, 1., dll.)
4. Split teks menjadi chunks berdasarkan pasal yang ditemukan
5. Setiap chunk menyimpan metadata: nomor pasal, judul pasal, nama file, halaman
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pypdf

# Setup logging agar kita bisa lihat proses chunking di terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    Satu chunk = unit teks legal (bisa pasal atau ayat).

    Attributes:
        chunk_id     : ID unik, contoh "kontrak_pasal_1_ayat_2"
        text         : Isi teks chunk
        pasal_number : Nomor pasal
        pasal_title  : Judul pasal
        source_file  : Nama file asal
        page_hint    : Nomor halaman (opsional)
        metadata     : Info tambahan (ayat_number, level, dll)
    """
    chunk_id: str
    text: str
    pasal_number: int
    pasal_title: str
    source_file: str
    page_hint: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Konversi ke dictionary untuk disimpan ke Qdrant.
        Metadata penting diexpose secara eksplisit agar mudah difilter.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "pasal_number": self.pasal_number,
            "pasal_title": self.pasal_title,
            "source": self.source_file,
            "page_hint": self.page_hint,

            # 🔥 metadata penting untuk retrieval
            "ayat_number": self.metadata.get("ayat_number"),
            "level": self.metadata.get("level", "pasal"),

            # 🔁 optional: simpan metadata lain kalau ada
            "extra_metadata": {
                k: v for k, v in self.metadata.items()
                if k not in ["ayat_number", "level"]
            }
        }


# ─── Pola Regex untuk mendeteksi awal pasal ────────────────────────────────────

# Kita dukung berbagai format penulisan pasal dalam kontrak Indonesia/Inggris:
# - "Pasal 1" / "PASAL 1"
# - "Pasal I" / "PASAL I" (angka romawi)
# - "Article 1" / "ARTICLE 1"
# - "1." di awal baris dengan huruf kapital setelahnya
PASAL_PATTERNS = [
    # Format: "Pasal 1" atau "PASAL 1" dengan judul opsional di baris yang sama
    r"(?i)^(pasal\s+(\d+|[IVXivx]+))\s*[:\-–]?\s*(.*)",
    # Format: "Article 1" atau "ARTICLE 1"
    r"(?i)^(article\s+(\d+|[IVXivx]+))\s*[:\-–]?\s*(.*)",
    # Format: "BAB I" atau "BAB 1"
    r"(?i)^(bab\s+(\d+|[IVXivx]+))\s*[:\-–]?\s*(.*)",
]

AYAT_PATTERN = re.compile(
    r"(?i)(ayat\s+(\d+))\s*[.:]?\s*(.*?)(?=(ayat\s+\d+)|$)",
    re.DOTALL
)

# Gabungkan semua pattern menjadi satu regex dengan OR
COMBINED_PATTERN = re.compile(
    "|".join(PASAL_PATTERNS),
    re.MULTILINE
)

# Romawi ke integer — untuk normalisasi nomor pasal
ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
}


def roman_to_int(s: str) -> Optional[int]:
    """Konversi angka romawi ke integer. Return None jika bukan angka romawi."""
    return ROMAN_MAP.get(s.upper())


def parse_pasal_number(raw: str) -> int:
    """
    Dari string seperti '1', 'I', 'IV', ekstrak nomor pasal sebagai integer.
    Jika gagal, return 0.
    """
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    roman = roman_to_int(raw)
    if roman is not None:
        return roman
    return 0


# ─── Fungsi Utama ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Baca semua halaman dari PDF dan gabungkan menjadi satu string teks.
    
    Args:
        pdf_path: Path ke file PDF
        
    Returns:
        String teks gabungan dari semua halaman
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
        ValueError: Jika PDF tidak bisa dibaca atau kosong
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")

    logger.info(f"Membaca PDF: {pdf_path.name}")
    
    text_pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        
        if len(reader.pages) == 0:
            raise ValueError(f"PDF kosong: {pdf_path.name}")
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
            
    full_text = "\n".join(text_pages)
    logger.info(f"  → {len(reader.pages)} halaman, {len(full_text)} karakter diekstrak")
    
    if not full_text.strip():
        raise ValueError(
            f"Tidak ada teks yang bisa diekstrak dari {pdf_path.name}. "
            "Mungkin PDF ini hasil scan — gunakan ocr.py terlebih dahulu."
        )
    
    return full_text


def extract_preamble_entities(text: str) -> dict:
    """
    Extract:
    - Judul kontrak
    - Pihak pertama
    - Pihak kedua
    """

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 🔥 Judul = biasanya baris pertama ALL CAPS
    doc_title = None
    for line in lines[:5]:
        if line.isupper() and len(line) > 10:
            doc_title = line
            break

    # 🔥 Pihak pertama & kedua
    pihak_pertama = None
    pihak_kedua = None

    pihak_pattern = re.compile(
    r"Nama\s*:\s*(.*?)\s+Jabatan\s*:\s*(.*?)\s+Selanjutnya disebut sebagai\s*PIHAK\s*(PERTAMA|KEDUA)",
    re.IGNORECASE | re.DOTALL
    )

    matches = pihak_pattern.findall(text)

    for m in matches:
        nama = m[0].strip()
        label = m[2].upper()

        if "PERTAMA" in label:
            pihak_pertama = nama
        elif "KEDUA" in label:
            pihak_kedua = nama

    return {
        "doc_title": doc_title,
        "pihak_pertama": pihak_pertama,
        "pihak_kedua": pihak_kedua,
    }

def split_by_pasal(text: str, source_file: str) -> list[Chunk]:
    """
    Split teks menjadi chunks berdasarkan PASAL → AYAT (lebih granular).

    Flow:
    - Preamble
    - Pasal
        - Ayat (jadi chunk utama)

    Jika tidak ada ayat → fallback ke pasal-level chunk
    """

    chunks: list[Chunk] = []

    matches = list(COMBINED_PATTERN.finditer(text))

    if not matches:
        logger.warning(
            f"⚠️ Tidak ditemukan pasal di '{source_file}', fallback full document."
        )
        chunks.append(Chunk(
            chunk_id=f"{Path(source_file).stem}_preamble",
            text=text.strip(),
            pasal_number=0,
            pasal_title="Preamble / Full Document",
            source_file=source_file,
            metadata={"level": "document"}
        ))
        return chunks

    # PREAMBLE
    preamble_text = text[:matches[0].start()].strip()

    if preamble_text:
        entities = extract_preamble_entities(preamble_text)

        chunks.append(Chunk(
            chunk_id=f"{Path(source_file).stem}_preamble",
            text=preamble_text,
            pasal_number=0,
            pasal_title="Preamble",
            source_file=source_file,
            metadata={
                "level": "preamble",
                **entities
            }
        ))

    stem = Path(source_file).stem.lower().replace(" ", "_")

    # LOOP PASAL
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        # Extract pasal number + title
        groups = match.groups()
        raw_num = None
        raw_title = ""

        step = 3
        for j in range(0, len(groups), step):
            if groups[j] is not None:
                raw_num = groups[j + 1]
                raw_title = groups[j + 2] or ""
                break

        pasal_num = parse_pasal_number(raw_num) if raw_num else i + 1
        pasal_title = raw_title.strip().strip(":-–").strip() or f"Pasal {pasal_num}"

        # 🔥 SPLIT KE AYAT
        ayat_matches = list(AYAT_PATTERN.finditer(chunk_text))

        if ayat_matches:
            for ayat_match in ayat_matches:
                ayat_num = int(ayat_match.group(2))
                ayat_text = ayat_match.group(0).strip()

                chunk_id = f"{stem}_pasal_{pasal_num}_ayat_{ayat_num}"

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=ayat_text,
                    pasal_number=pasal_num,
                    pasal_title=pasal_title,
                    source_file=source_file,
                    metadata={
                        "ayat_number": ayat_num,
                        "level": "ayat"
                    }
                ))

        else:
            # fallback kalau tidak ada ayat
            chunk_id = f"{stem}_pasal_{pasal_num}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                pasal_number=pasal_num,
                pasal_title=pasal_title,
                source_file=source_file,
                metadata={
                    "level": "pasal"
                }
            ))

        logger.debug(
            f"Chunk PASAL {pasal_num} → "
            f"{len(ayat_matches)} ayat" if ayat_matches else "no ayat"
        )

    logger.info(f"→ {len(chunks)} chunks berhasil dibuat dari '{source_file}'")
    return chunks


def chunk_pdf(pdf_path: str | Path) -> list[Chunk]:
    """
    Entry point utama: Baca PDF dan kembalikan list of Chunk.
    
    Args:
        pdf_path: Path ke file PDF (bisa string atau Path object)
        
    Returns:
        List of Chunk, siap untuk di-embed dan disimpan ke vector DB
        
    Example:
        chunks = chunk_pdf("data/dummy/Kontrak_Sewa.pdf")
        for c in chunks:
            print(c.chunk_id, "→", c.pasal_title)
    """
    pdf_path = Path(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = split_by_pasal(text, source_file=pdf_path.name)
    return chunks


def chunk_directory(directory: str | Path) -> list[Chunk]:
    """
    Proses semua file PDF di dalam sebuah direktori sekaligus.
    
    Args:
        directory: Path ke folder yang berisi file-file PDF
        
    Returns:
        List of Chunk dari semua PDF yang ditemukan
        
    Example:
        all_chunks = chunk_directory("data/dummy/")
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Bukan direktori: {directory}")
    
    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"Tidak ada file PDF ditemukan di: {directory}")
        return []
    
    all_chunks: list[Chunk] = []
    for pdf_file in pdf_files:
        try:
            chunks = chunk_pdf(pdf_file)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Gagal memproses {pdf_file.name}: {e}")
    
    logger.info(f"Total: {len(all_chunks)} chunks dari {len(pdf_files)} file PDF")
    return all_chunks


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Jalankan dengan: python ingestion/chunker.py
    Akan memproses semua PDF di folder data/dummy/
    """
    import sys
    
    data_dir = Path("data/dummy")
    if not data_dir.exists():
        logger.error(f"Folder {data_dir} tidak ditemukan. Jalankan dari root project.")
        sys.exit(1)
    
    chunks = chunk_directory(data_dir)
    
    print("\n" + "="*60)
    print(f"HASIL CHUNKING — {len(chunks)} chunks total")
    print("="*60)
    for c in chunks:
        print(f"  [{c.chunk_id}]")
        print(f"    Pasal   : {c.pasal_number}")
        print(f"    Judul   : {c.pasal_title}")
        print(f"    Sumber  : {c.source_file}")
        print(f"    Preview : {c.text[:80].replace(chr(10), ' ')}...")
        print()