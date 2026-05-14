import os
from dotenv import load_dotenv
from groq import Groq
import threading
from typing import List

from app.schemas.hadits_schema import HaditsResultForSummarizer, LLMSummarizerRequest

load_dotenv()

# Memuat GROQ_API_KEY_1 sampai GROQ_API_KEY_4
_api_keys: List[str] = [
    os.getenv(f"GROQ_API_KEY_{i}")
    for i in range(1, 5)
    if os.getenv(f"GROQ_API_KEY_{i}") is not None
]

if not _api_keys:
    raise ValueError("Tidak ada satupun GROQ_API_KEY yang ditemukan di file .env")

_api_key_index = 0
_api_key_lock = threading.Lock()

def get_next_api_key() -> str:
    global _api_key_index
    with _api_key_lock:
        key = _api_keys[_api_key_index]
        _api_key_index = (_api_key_index + 1) % len(_api_keys)
        return key

def summarize_hadits(request: LLMSummarizerRequest) -> str:
    top_10_hadits = request.hadits_results[:10]

    api_key = get_next_api_key()
    client = Groq(api_key=api_key)

    # Format data konteks dengan rapi (Gunakan top_10_hadits, bukan request.hadits_results)
    hadits_text = "\n\n".join(
        [
            f"Perawi: {h.nama_perawi} | No. {h.nomor_hadits}\nTeks: {h.terjemahan}"
            for i, h in enumerate(top_10_hadits)
        ]
    )

    # ==========================================
    # PROMPT ENGINEERING SECTION
    # ==========================================
    
    system_prompt = """Anda adalah asisten AI yang ahli, objektif, dan teliti dalam merangkum literatur Islam (Hadits). Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan hadits-hadits yang diberikan.

IKUTI ATURAN KETAT INI:

1. EVALUASI RELEVANSI (ANTI-MAKSA):
   Sebelum merangkum, nilai apakah hadits yang diberikan benar-benar relevan dengan pertanyaan. Jika tidak ada yang relevan, tulis satu kalimat jujur seperti: "Hadits-hadits yang ditemukan belum secara langsung membahas topik ini." Lalu rangkum secara singkat apa yang sebenarnya dibahas hadits-hadits tersebut.

2. SINTESIS NARATIF — BUKAN DAFTAR DOKUMEN:
   DILARANG menyebut "Dokumen 1", "Dokumen 2", dst. Gabungkan isi hadits menjadi satu narasi koheren yang mengalir. Tulis seolah Anda menjelaskan kepada pembaca awam yang ingin memahami ajaran, bukan kepada peneliti yang menelusuri sumber.

3. REFERENSI SUMBER YANG NATURAL:
   Jika perlu menyebut sumber, gunakan nama perawi dan nomor hadits saja — contoh: "(HR. Bukhari no. 6116)" atau "dalam riwayat Muslim no. 2607". Letakkan di akhir kalimat atau paragraf yang relevan, bukan di awal. Jangan sebut nomor dokumen internal.

4. FORMAT YANG ELEGAN:
   - Paragraf pembuka: satu kalimat padat yang langsung menjawab atau merangkum inti ajaran.
   - Gunakan bullet points (-) untuk poin-poin hikmah atau hukum yang berbeda.
   - Gunakan **bold** untuk konsep atau istilah kunci saja, jangan berlebihan.

5. RINGKAS DAN BERBOBOT:
   Tidak perlu panjang. Satu paragraf pembuka + 2-4 bullet points sudah cukup. Fokus pada substansi ajaran, bukan pada metadatanya."""

    user_prompt = f"""Pertanyaan Pengguna: "{request.query}"

Hadits-hadits yang ditemukan:
{hadits_text}

Buatlah ringkasan sesuai aturan. Ingat: jangan sebut "Dokumen N" — gunakan nama perawi dan nomor hadits jika perlu merujuk sumber."""

    # Memanggil Groq API menggunakan model Llama 3
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2, 
        max_tokens=1024,
    )

    return response.choices[0].message.content