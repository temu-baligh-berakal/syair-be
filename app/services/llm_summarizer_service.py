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
    top_3_hadits = request.hadits_results[:3]

    api_key = get_next_api_key()
    client = Groq(api_key=api_key)

    # Format data konteks dengan rapi (Gunakan top_3_hadits, bukan request.hadits_results)
    hadits_text = "\n\n".join(
        [
            f"[Dokumen {i+1}] - Perawi: {h.nama_perawi}\nTeks: {h.terjemahan}"
            for i, h in enumerate(top_3_hadits)
        ]
    )

    # ==========================================
    # PROMPT ENGINEERING SECTION
    # ==========================================
    
    system_prompt = """Anda adalah asisten AI yang ahli, objektif, dan teliti dalam merangkum literatur Islam (Hadits). Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan [Dokumen] yang diberikan.

IKUTI ATURAN KETAT INI:
1. EVALUASI RELEVANSI (ANTI-MAKSA): Sebelum merangkum, nilai apakah dokumen yang diberikan benar-benar menjawab atau relevan dengan pertanyaan pengguna. Jika dokumen tidak relevan, JANGAN memaksakan hubungan atau mengarang ajaran. Cukup katakan: "Berdasarkan hadits yang ditemukan, tidak ada informasi yang secara langsung dan spesifik menjawab pertanyaan ini." lalu berikan ringkasan singkat tentang apa yang sebenarnya dibahas dalam dokumen tersebut.
2. SINTESIS, BUKAN MENGULANG: Jangan merangkum dokumen satu per satu (misal: "Dokumen 1 mengatakan... Dokumen 2 mengatakan..."). Gabungkan intisari hukum, hikmah, atau ajaran dari dokumen-dokumen tersebut menjadi satu narasi yang koheren.
3. FORMAT YANG ELEGAN: Gunakan Markdown untuk menstrukturkan jawaban Anda. Gunakan paragraf pembuka yang padat, lalu gunakan *bullet points* (-) untuk poin-poin utama, dan gunakan huruf tebal (**bold**) untuk menekankan konsep atau istilah kunci.
4. RINGKAS: Buat ringkasan yang to-the-point. Jangan sebutkan nomor hadits atau nama perawi di dalam teks ringkasan (fokus pada substansi ajarannya saja)."""

    user_prompt = f"""Pertanyaan Pengguna: "{request.query}"

Konteks Dokumen Hadits yang Ditemukan:
{hadits_text}

Buatlah ringkasan berdasarkan aturan yang telah ditetapkan."""

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
        temperature=0.2, # Diturunkan ke 0.2 agar lebih analitis, kaku pada fakta, dan tidak berhalusinasi
        max_tokens=1024,
    )

    return response.choices[0].message.content