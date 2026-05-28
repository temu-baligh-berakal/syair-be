import os
import logging
from dotenv import load_dotenv
from groq import Groq
import threading
from typing import List

from app.schemas.hadits_schema import (
    HaditsResultForSummarizer,
    LLMSummarizerRequest,
)

load_dotenv()

logger = logging.getLogger(__name__)

class SummarizerError(Exception):
    """Kesalahan terkontrol untuk service rangkuman."""


def _load_api_keys() -> List[str]:
    return [
        os.getenv(f"GROQ_API_KEY_{i}")
        for i in range(1, 5)
        if os.getenv(f"GROQ_API_KEY_{i}") is not None
    ]


_api_keys: List[str] = _load_api_keys()

_api_key_index = 0
_api_key_lock = threading.Lock()


def get_next_api_key() -> str:
    global _api_key_index

    if not _api_keys:
        raise SummarizerError(
            "GROQ API key belum dikonfigurasi di backend."
        )

    with _api_key_lock:
        key = _api_keys[_api_key_index]
        current_index = _api_key_index
        _api_key_index = (_api_key_index + 1) % len(_api_keys)
        
        # Log rotasi kunci (hanya tampilkan 4 karakter awal agar aman)
        logger.info(f"LLM Key Rotation: Menggunakan API Key index {current_index} (Prefix: {key[:8]}...)")
        
        return key


def summarize_hadits(request: LLMSummarizerRequest) -> str:
    top_10_hadits = request.hadits_results[:10]

    # Format konteks hadits
    hadits_text = "\n\n".join(
        [
            f"Perawi: {h.nama_perawi} | No. {h.nomor_hadits}\nTeks: {h.terjemahan}"
            for i, h in enumerate(top_10_hadits)
        ]
    )

    # ==========================================
    # PROMPT ENGINEERING SECTION
    # ==========================================

    system_prompt = """Anda adalah asisten AI yang ahli, objektif, dan teliti dalam merangkum literatur Islam (Hadits). 
Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan hadits-hadits yang diberikan.

IKUTI ATURAN KETAT INI:

1. EVALUASI RELEVANSI (ANTI-MAKSA):
   Sebelum merangkum, nilai apakah hadits yang diberikan benar-benar relevan dengan pertanyaan.
   Jika tidak ada yang relevan, tulis secara jujur:
   "Hadits-hadits yang ditemukan belum secara langsung membahas topik ini."
   Lalu rangkum secara singkat apa yang sebenarnya dibahas.

2. SINTESIS NARATIF — BUKAN DAFTAR DOKUMEN:
   DILARANG menyebut "Dokumen 1", "Dokumen 2", dst. Gabungkan isi hadits menjadi satu narasi koheren yang mengalir. Tulis seolah Anda menjelaskan kepada pembaca awam yang ingin memahami ajaran, bukan kepada peneliti yang menelusuri sumber.

3. REFERENSI SUMBER YANG NATURAL:
   Jika perlu menyebut sumber, gunakan format alami seperti:
   "(HR. Bukhari no. 6116)" atau "dalam riwayat Muslim no. 2607".
   Letakkan di akhir kalimat atau paragraf yang relevan, bukan di awal. Jangan menyebut "Dokumen 1", "Dokumen 2", dan seterusnya dalam jawaban akhir.

4. FORMAT YANG ELEGAN:
   - Awali dengan paragraf pembuka singkat yang langsung menjawab atau merangkum inti ajaran.
   - Gunakan bullet points (-) untuk poin-poin hikmah atau hukum yang berbeda atau penting.
   - Gunakan **bold** untuk konsep atau istilah kunci utama saja, jangan berlebihan.

5. RINGKAS DAN BERBOBOT:
   Fokus pada substansi ajaran. Tidak perlu panjang. Satu paragraf pembuka + 2-4 bullet points sudah cukup. Fokus pada substansi ajaran, bukan pada metadatanya.

6. JAGA INTENT PERTANYAAN:
   Jika pengguna bertanya tentang cara, langkah, tata cara, atau prosedur, maka jawaban HARUS mengikuti intent tersebut.
   Jika hadits hanya menjelaskan hukum, kewajiban, sebab, keutamaan, tetapi TIDAK menjelaskan langkah-langkah secara runtut, maka katakan dengan jelas bahwa:
   "Hadits yang ditemukan lebih banyak membahas hukum atau keutamaannya, bukan tata caranya secara rinci."

7. DILARANG MENAMBAHKAN LANGKAH YANG TIDAK ADA:
   Jangan membuat urutan tata cara atau prosedur yang tidak eksplisit di hadits.
"""

    user_prompt = f"""Pertanyaan Pengguna:
"{request.query}"

Hadits-hadits yang ditemukan:
{hadits_text}

Buatlah ringkasan sesuai aturan di atas. Ingat: jangan sebut "Dokumen N" — gunakan nama perawi dan nomor hadits jika perlu menyebut sumber.
"""

    max_retries = len(_api_keys)
    last_exception = None

    for attempt in range(max_retries):
        api_key = get_next_api_key()
        client = Groq(api_key=api_key)
        current_key_index = (_api_key_index - 1) % max_retries

        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Memulai peringkasan (Key Index: {current_key_index})")
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=1024,
            )
            
            content = response.choices[0].message.content if response.choices else None
            if not content or not content.strip():
                raise SummarizerError("Layanan LLM mengembalikan ringkasan kosong.")

            logger.info(f"Peringkasan berhasil pada attempt {attempt + 1}.")
            return content

        except Exception as exc:
            last_exception = exc
            error_msg = str(exc)
            
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                logger.warning(f"RATE LIMIT on Key Index {current_key_index}. Retrying ({attempt + 1}/{max_retries})...")
                continue
            
            logger.error(f"Error pada attempt {attempt + 1}: {error_msg}")
            # Jika bukan rate limit, tetap coba kunci berikutnya sampai max_retries
            if attempt < max_retries - 1:
                continue
            break

    raise SummarizerError(f"Gagal setelah {max_retries} percobaan. Error terakhir: {str(last_exception)}")