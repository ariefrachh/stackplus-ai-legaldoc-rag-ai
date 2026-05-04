"""
templates/__init__.py — Template klausul berbahaya untuk risk detection

Apa itu template klausul berbahaya?
  Ini adalah kumpulan contoh teks yang merepresentasikan jenis-jenis klausul
  yang sering merugikan pihak yang lebih lemah (misalnya: karyawan, penyewa, klien kecil).

  Cara kerjanya di detector.py nanti:
  1. Ambil setiap pasal dari kontrak
  2. Hitung cosine similarity antara embedding pasal tersebut dengan embedding setiap template
  3. Jika similarity > threshold → pasal tersebut di-flag sebagai Red Flag / Yellow Flag

Mengapa pakai banyak contoh per klausul?
  Satu klausul berbahaya bisa ditulis dalam banyak cara berbeda.
  Dengan beberapa contoh, embedding template menjadi lebih "robust" 
  (menangkap berbagai variasi bahasa).
  
  Kita rata-rata semua embedding template dalam satu kategori → 
  hasilnya satu "centroid" vector yang merepresentasikan seluruh kategori.
"""

from dataclasses import dataclass, field


@dataclass
class RiskTemplate:
    """
    Representasi satu kategori klausul berbahaya.
    
    Attributes:
        name       : Nama kategori (untuk ditampilkan ke user)
        risk_level : "RED" (sangat berbahaya) atau "YELLOW" (perlu perhatian)
        description: Penjelasan mengapa klausul ini berbahaya
        templates  : List of teks contoh klausul berbahaya (makin banyak makin baik)
        advice     : Saran untuk user jika klausul ini ditemukan
    """
    name: str
    risk_level: str  # "RED" atau "YELLOW"
    description: str
    templates: list[str]
    advice: str
    tags: list[str] = field(default_factory=list)


# ─── Daftar Template Klausul Berbahaya ─────────────────────────────────────────
#
# Setiap template berisi beberapa contoh teks klausul.
# Contoh-contoh ini akan di-embed dan dirata-rata untuk membentuk "centroid".
#
# Sumber referensi:
# - CUAD (Contract Understanding Atticus Dataset)
# - Praktik hukum kontrak Indonesia (KUHPerdata, UU Ketenagakerjaan)
# ─────────────────────────────────────────────────────────────────────────────

RISK_TEMPLATES: list[RiskTemplate] = [

    # ── RED FLAGS ──────────────────────────────────────────────────────────────

    RiskTemplate(
        name="Indemnifikasi Sepihak",
        risk_level="RED",
        description=(
            "Klausul yang mewajibkan satu pihak menanggung SEMUA kerugian, biaya, "
            "dan tuntutan hukum — termasuk yang bukan kesalahannya. "
            "Ini sangat merugikan karena eksposur finansial bisa tak terbatas."
        ),
        templates=[
            "Pihak Kedua wajib membebaskan, membela, dan mengganti rugi Pihak Pertama dari dan terhadap segala klaim, kerugian, denda, biaya, dan pengeluaran apapun.",
            "The second party shall indemnify, defend, and hold harmless the first party from any and all claims, damages, losses, costs, and expenses of any nature whatsoever.",
            "Pihak Kedua bertanggung jawab sepenuhnya atas seluruh kerugian yang timbul, baik secara langsung maupun tidak langsung, tanpa batas.",
            "Vendor shall indemnify and hold the company harmless from all liabilities, damages, costs, and attorney's fees arising from any claim.",
            "Penyedia layanan membebaskan klien dari segala tuntutan, kerugian, dan biaya hukum tanpa pengecualian.",
        ],
        advice=(
            "Negosiasikan agar indemnifikasi bersifat TIMBAL BALIK (mutual) dan dibatasi "
            "hanya untuk kerugian yang secara langsung disebabkan oleh kelalaian pihak tersebut. "
            "Tambahkan 'cap' (batas maksimum) tanggung jawab."
        ),
        tags=["indemnifikasi", "ganti rugi", "tanggung jawab tak terbatas"],
    ),

    RiskTemplate(
        name="Perpanjangan Otomatis (Auto-Renewal)",
        risk_level="RED",
        description=(
            "Kontrak otomatis diperpanjang jika tidak ada notifikasi penghentian "
            "dalam waktu tertentu. Berbahaya karena user bisa terjebak dalam kontrak "
            "yang tidak lagi diinginkan, seringkali dengan syarat baru yang lebih memberatkan."
        ),
        templates=[
            "Kontrak ini akan diperpanjang secara otomatis selama 1 (satu) tahun berikutnya kecuali salah satu pihak memberikan pemberitahuan tertulis minimal 90 hari sebelum berakhirnya masa kontrak.",
            "This agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least 60 days prior to expiration.",
            "Perjanjian ini diperpanjang secara otomatis apabila tidak ada pemberitahuan pengakhiran 3 bulan sebelum jatuh tempo.",
            "The contract will be automatically extended unless terminated with 90 days written notice before the end of each term.",
            "Masa berlaku perjanjian diperpanjang demi hukum apabila tidak ada pernyataan tertulis untuk mengakhiri paling lambat 6 bulan sebelum berakhir.",
        ],
        advice=(
            "Pastikan kamu punya sistem reminder untuk tenggat waktu notifikasi. "
            "Negosiasikan agar auto-renewal dikurangi menjadi opt-in (harus aktif konfirmasi), "
            "atau perpendek periode notifikasi menjadi 30 hari."
        ),
        tags=["auto-renewal", "perpanjangan otomatis", "rollover"],
    ),

    RiskTemplate(
        name="Perubahan Sepihak (Unilateral Modification)",
        risk_level="RED",
        description=(
            "Satu pihak (biasanya yang lebih kuat) bisa mengubah syarat kontrak "
            "kapan saja tanpa persetujuan pihak lain. Ini membuat kontrak tidak punya "
            "kepastian hukum bagi pihak yang lebih lemah."
        ),
        templates=[
            "Pihak Pertama berhak sewaktu-waktu mengubah, menambah, atau menghapus ketentuan dalam perjanjian ini tanpa persetujuan Pihak Kedua.",
            "The company reserves the right to modify, amend, or update the terms of this agreement at any time at its sole discretion without prior notice.",
            "Kami dapat mengubah syarat dan ketentuan ini kapan saja. Perubahan berlaku segera setelah dipublikasikan.",
            "The service provider may change the terms of service at any time, and continued use of the service constitutes acceptance.",
            "Manajemen berhak mengubah ketentuan perjanjian ini secara sepihak berdasarkan kebijakan perusahaan.",
        ],
        advice=(
            "Klausul ini seharusnya tidak ada atau diubah menjadi: perubahan hanya bisa dilakukan "
            "dengan persetujuan tertulis KEDUA BELAH PIHAK. "
            "Jika klien menolak, minta minimal ada kewajiban notifikasi 30-60 hari sebelum perubahan berlaku."
        ),
        tags=["perubahan sepihak", "unilateral", "amendment"],
    ),

    RiskTemplate(
        name="Klausul Non-Kompetisi Berlebihan",
        risk_level="RED",
        description=(
            "Membatasi seseorang bekerja di industri yang sama setelah kontrak berakhir, "
            "dengan cakupan yang terlalu luas (durasi >1 tahun, area geografis luas, "
            "atau industri yang sangat umum). Di Indonesia, klausul ini sering tidak valid "
            "secara hukum tapi tetap menakutkan secara psikologis."
        ),
        templates=[
            "Karyawan tidak boleh bekerja, mendirikan, atau memiliki kepentingan dalam perusahaan yang bergerak di bidang yang sama selama 3 tahun setelah berakhirnya hubungan kerja.",
            "Employee agrees not to engage in any business competitive with the company for a period of 2 years following termination, within any country where the company operates.",
            "Selama 2 tahun setelah pengunduran diri, karyawan dilarang bekerja di perusahaan kompetitor manapun di wilayah Asia Tenggara.",
            "The employee shall not directly or indirectly compete with the company for 36 months post-employment in any capacity.",
        ],
        advice=(
            "Di Indonesia, klausul non-compete harus reasonable: max 1 tahun, area terbatas, "
            "industri yang sangat spesifik. Klausul yang terlalu luas bisa diuji di pengadilan. "
            "Konsultasikan dengan lawyer untuk menilai validitasnya."
        ),
        tags=["non-kompetisi", "non-compete", "larangan bekerja"],
    ),

    RiskTemplate(
        name="Yurisdiksi Asing / Hukum Asing",
        risk_level="RED",
        description=(
            "Sengketa diselesaikan di pengadilan negara asing atau menggunakan hukum negara asing. "
            "Ini sangat memberatkan karena biaya litigasi di luar negeri sangat mahal "
            "dan prosesnya sulit diakses bagi perusahaan Indonesia kecil."
        ),
        templates=[
            "Segala sengketa yang timbul dari perjanjian ini akan diselesaikan di Pengadilan Singapura dengan menerapkan hukum Singapura.",
            "This agreement shall be governed by and construed in accordance with the laws of England and Wales. Any disputes shall be submitted to the exclusive jurisdiction of the English courts.",
            "Perjanjian ini tunduk pada hukum Amerika Serikat, Negara Bagian Delaware, dan para pihak tunduk pada yurisdiksi eksklusif pengadilan Delaware.",
            "All disputes arising from this contract shall be resolved by arbitration in Hong Kong under the HKIAC rules.",
        ],
        advice=(
            "Negosiasikan agar sengketa diselesaikan di Indonesia, di bawah hukum Indonesia "
            "dan/atau arbitrase BANI (Badan Arbitrase Nasional Indonesia). "
            "Jika tidak bisa dihindari, pastikan ada 'governing law' yang menguntungkan."
        ),
        tags=["yurisdiksi asing", "hukum asing", "arbitrase luar negeri"],
    ),

    RiskTemplate(
        name="Pengalihan Kontrak Sepihak",
        risk_level="RED",
        description=(
            "Satu pihak bisa mengalihkan (assign) kontrak ke pihak ketiga tanpa persetujuan. "
            "Berbahaya karena kamu bisa tiba-tiba berurusan dengan entitas yang berbeda "
            "dari yang kamu sepakati."
        ),
        templates=[
            "Pihak Pertama berhak mengalihkan seluruh hak dan kewajiban dalam perjanjian ini kepada pihak manapun tanpa perlu persetujuan Pihak Kedua.",
            "The company may assign this agreement and its rights hereunder to any successor, affiliate, or third party without the consent of the other party.",
            "Hak dan kewajiban dalam perjanjian ini dapat dialihkan oleh Pihak Pertama kepada pihak lain tanpa pemberitahuan sebelumnya.",
        ],
        advice=(
            "Tambahkan klausul: pengalihan kontrak hanya diperbolehkan dengan persetujuan "
            "tertulis dari kedua pihak, kecuali dalam kasus merger/akuisisi perusahaan."
        ),
        tags=["assignment", "pengalihan", "novasi"],
    ),

    # ── YELLOW FLAGS ────────────────────────────────────────────────────────────

    RiskTemplate(
        name="Denda Keterlambatan Tinggi",
        risk_level="YELLOW",
        description=(
            "Denda keterlambatan pembayaran yang melebihi 2% per bulan atau yang "
            "bersifat kumulatif tanpa batas atas bisa menjadi sangat memberatkan "
            "jika ada keterlambatan yang tidak disengaja."
        ),
        templates=[
            "Keterlambatan pembayaran dikenakan denda sebesar 5% per bulan dari jumlah yang tertunggak.",
            "Late payment shall incur a penalty of 3% per month compounded monthly on the outstanding amount.",
            "Pihak Kedua wajib membayar denda keterlambatan sebesar 2% per hari dari nilai tagihan.",
            "A late fee of 1.5% per month will be charged on any overdue balance.",
            "Biaya keterlambatan dihitung secara harian dan diakumulasi tanpa batas maksimum.",
        ],
        advice=(
            "Denda keterlambatan wajar di Indonesia adalah maksimal 2% per bulan. "
            "Jika melebihi itu, negosiasikan ke bawah. Pastikan ada 'grace period' "
            "(misal 7-14 hari) sebelum denda mulai berjalan."
        ),
        tags=["denda", "penalti", "keterlambatan bayar"],
    ),

    RiskTemplate(
        name="Klausul Force Majeure Sempit",
        risk_level="YELLOW",
        description=(
            "Force majeure (keadaan kahar) yang definisinya terlalu sempit atau "
            "tidak mencakup kejadian yang wajar seperti pandemi, bencana alam, "
            "atau gangguan pemerintah. Ini bisa membuat kamu tetap bertanggung jawab "
            "meski kejadian di luar kendali."
        ),
        templates=[
            "Keadaan kahar hanya mencakup bencana alam berupa gempa bumi, banjir, dan kebakaran yang telah ditetapkan oleh pemerintah.",
            "Force majeure events are limited to acts of God, war, and government prohibition only.",
            "Perjanjian ini tidak dapat diakhiri atau ditangguhkan kecuali atas dasar keadaan kahar yang diakui secara resmi oleh pemerintah.",
        ],
        advice=(
            "Force majeure seharusnya mencakup: bencana alam, pandemi, perang, kebijakan pemerintah, "
            "pemadaman listrik masif, dan kejadian lain di luar kendali wajar. "
            "Minta agar definisi diperluas dan ada prosedur notifikasi yang jelas."
        ),
        tags=["force majeure", "keadaan kahar", "act of god"],
    ),

    RiskTemplate(
        name="Pembatasan Kewajiban Sepihak",
        risk_level="YELLOW",
        description=(
            "Hanya satu pihak yang dibatasi tanggung jawabnya, sementara pihak lain "
            "tidak ada batasnya. Ini menciptakan ketidakseimbangan risiko."
        ),
        templates=[
            "Total tanggung jawab Pihak Pertama tidak melebihi nilai 1 bulan pembayaran terakhir.",
            "The company's liability shall not exceed the amount paid in the last 30 days.",
            "Liability of the service provider is capped at USD 100 under any circumstances.",
            "Tanggung jawab penyedia layanan dibatasi maksimal senilai biaya langganan 1 bulan.",
        ],
        advice=(
            "Pastikan pembatasan kewajiban bersifat TIMBAL BALIK (mutual). "
            "Jika cap berlaku untuk vendor, cap juga harus berlaku untuk klien. "
            "Nilai cap seharusnya proporsional dengan nilai kontrak."
        ),
        tags=["liability cap", "pembatasan tanggung jawab"],
    ),

    RiskTemplate(
        name="Hak Kekayaan Intelektual Berlebihan",
        risk_level="YELLOW",
        description=(
            "Perusahaan mengklaim kepemilikan atas SEMUA karya yang dibuat karyawan/freelancer, "
            "termasuk proyek pribadi yang dibuat di luar jam kerja dan tidak menggunakan "
            "sumber daya perusahaan."
        ),
        templates=[
            "Segala karya, penemuan, inovasi, atau kekayaan intelektual yang dihasilkan karyawan selama masa kerja adalah milik eksklusif perusahaan.",
            "All inventions, works, and developments created by the employee during the term of employment, whether during or outside of working hours, shall be the sole property of the company.",
            "Seluruh hasil kerja, termasuk yang dibuat di luar jam kerja, menjadi hak milik perusahaan.",
        ],
        advice=(
            "Klausul HKI seharusnya terbatas pada: (1) karya yang dibuat dalam lingkup pekerjaan, "
            "(2) menggunakan sumber daya perusahaan, atau (3) terkait langsung dengan bisnis perusahaan. "
            "Proyek pribadi di luar jam kerja tanpa sumber daya perusahaan seharusnya milik kamu."
        ),
        tags=["hak cipta", "HAKI", "intellectual property", "IP"],
    ),

    RiskTemplate(
        name="Klausul Terminasi Sepihak Tanpa Alasan",
        risk_level="YELLOW",
        description=(
            "Satu pihak bisa mengakhiri kontrak kapan saja tanpa alasan (at-will termination) "
            "atau dengan periode notifikasi yang sangat singkat, meninggalkan pihak lain "
            "dalam kondisi sulit."
        ),
        templates=[
            "Pihak Pertama dapat mengakhiri perjanjian ini sewaktu-waktu tanpa memberikan alasan dengan pemberitahuan 7 hari.",
            "The company may terminate this agreement at any time for any reason or no reason with 24 hours notice.",
            "Kontrak dapat diputus oleh klien tanpa alasan dengan memberikan pemberitahuan 14 hari.",
        ],
        advice=(
            "Negosiasikan agar terminasi tanpa sebab (without cause) disertai kompensasi "
            "(misal: pembayaran sisa kontrak atau 3 bulan fee). "
            "Perpanjang periode notifikasi minimal 30 hari agar ada waktu mencari pengganti."
        ),
        tags=["terminasi", "PHK", "pengakhiran kontrak", "at-will"],
    ),
]


# ─── Utility ────────────────────────────────────────────────────────────────────

def get_templates_by_risk_level(level: str) -> list[RiskTemplate]:
    """Filter templates berdasarkan risk level ('RED' atau 'YELLOW')."""
    return [t for t in RISK_TEMPLATES if t.risk_level.upper() == level.upper()]


def get_template_by_name(name: str) -> RiskTemplate | None:
    """Cari template berdasarkan nama (case-insensitive)."""
    name_lower = name.lower()
    for t in RISK_TEMPLATES:
        if name_lower in t.name.lower():
            return t
    return None


def list_all_templates() -> None:
    """Print semua template yang tersedia."""
    print(f"\n{'='*60}")
    print(f"RISK TEMPLATES — {len(RISK_TEMPLATES)} klausul terdaftar")
    print(f"{'='*60}")
    for t in RISK_TEMPLATES:
        icon = "🔴" if t.risk_level == "RED" else "🟡"
        print(f"  {icon} [{t.risk_level}] {t.name}")
        print(f"       {len(t.templates)} contoh template")
    print()


if __name__ == "__main__":
    list_all_templates()