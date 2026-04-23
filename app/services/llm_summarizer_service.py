import os
from dotenv import load_dotenv
import google.generativeai as genai
import threading
from typing import List

from app.schemas.hadits_schema import HaditsResultForSummarizer, LLMSummarizerRequest

load_dotenv()

_api_keys: List[str] = [
    os.getenv(f"GEMINI_API_KEY_{i}")
    for i in range(1, 9)  # Assuming GEMINI_API_KEY_1 to GEMINI_API_KEY_8
    if os.getenv(f"GEMINI_API_KEY_{i}") is not None
]

_api_key_index = 0
_api_key_lock = threading.Lock()

def get_next_api_key() -> str:
    global _api_key_index
    with _api_key_lock:
        key = _api_keys[_api_key_index]
        _api_key_index = (_api_key_index + 1) % len(_api_keys)
        return key

def summarize_hadits(request: LLMSummarizerRequest) -> str:
    api_key = get_next_api_key()
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-3-flash-preview')

    hadits_text = "\n\n".join(
        [
            f"Hadits {h.referensi_lengkap} (Perawi: {h.nama_perawi}, Nomor: {h.nomor_hadits}):\n{h.terjemahan}"
            for h in request.hadits_results
        ]
    )

    prompt = f"""Ringkas dan sintesiskan hadits-hadits berikut yang berkaitan dengan query "{request.query}":

{hadits_text}

Berikan ringkasan yang koheren, padat, dan informatif berdasarkan semua hadits yang diberikan. Fokus pada inti ajaran atau konteks utama yang relevan dengan query.
"""

    response = model.generate_content(prompt)
    return response.text
