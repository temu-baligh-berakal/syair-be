import os
from dotenv import load_dotenv
from groq import Groq
import threading
from typing import List

from app.schemas.hadits_schema import HaditsResultForSummarizer, LLMSummarizerRequest

load_dotenv()

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
        raise SummarizerError("GROQ API key belum dikonfigurasi di backend.")

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
            f"[Dokumen {i+1}] - Perawi: {h.nama_perawi}\nTeks: {h.terjemahan}"
            for i, h in enumerate(top_10_hadits)
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
4. RINGKAS: Buat ringkasan yang to-the-point. Jangan sebutkan nomor hadits atau nama perawi di dalam teks ringkasan (fokus pada substansi ajarannya saja).
5. JAGA INTENT PERTANYAAN: Jika pengguna bertanya tentang **cara/langkah/tata cara/bagaimana melakukan sesuatu**, maka Anda WAJIB menjawab sesuai intent prosedural itu. Jika dokumen hanya menjelaskan **kapan**, **sebab**, **hukum**, atau **kewajiban**, tetapi tidak memberi langkah-langkah, maka katakan dengan tegas bahwa hasil yang ditemukan **lebih banyak membahas sebab atau kewajibannya, bukan tata caranya secara runtut**. Jangan mengubah jawaban menjadi penjelasan umum yang seolah-olah menjawab "cara".
6. JANGAN MENAMBAHKAN LANGKAH YANG TIDAK ADA: Untuk pertanyaan prosedural, jangan menulis urutan langkah kecuali langkah itu memang tampak jelas di dokumen yang diberikan."""

    user_prompt = f"""Pertanyaan Pengguna: "{request.query}"

Konteks Dokumen Hadits yang Ditemukan:
{hadits_text}

Buatlah ringkasan berdasarkan aturan yang telah ditetapkan."""

    try:
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
    except Exception as exc:
        raise SummarizerError(f"Gagal menghubungi layanan LLM: {str(exc)}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise SummarizerError("Layanan LLM mengembalikan ringkasan kosong.")

    return content
