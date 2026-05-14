import re

from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

from app.config import INDEX_NAME
from app.schemas.hadits_schema import HaditsResult, SearchResponse
from app.services.strategies import get_strategy

import app.services.strategies.knn     # noqa: F401
import app.services.strategies.bm25    # noqa: F401
import app.services.strategies.hybrid  # noqa: F401

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

def _parse_hit(hit: dict, score: float) -> HaditsResult:
    """Konversi satu hit OpenSearch menjadi HaditsResult dan ekstrak Highlight."""
    src = hit.get("_source", {})
    terjemahan_asli = src.get("terjemahan", "")
    
    # 1. Coba ambil highlight dari OpenSearch (Akan ada jika mode BM25 / Hybrid)
    highlights = hit.get("highlight", {}).get("terjemahan", [])
    
    if highlights:
        preview_raw = highlights[0]
        clean_preview = preview_raw.replace("**", "")
        
        preview = preview_raw
        
        # Tambahkan elipsis (...) di awal jika kutipan tidak dimulai dari awal kalimat
        if len(clean_preview) > 20 and not terjemahan_asli.startswith(clean_preview[:20]):
            preview = f"...{preview}"
            
        # Tambahkan elipsis (...) di akhir jika kutipan terpotong sebelum akhir kalimat
        if len(clean_preview) > 20 and not terjemahan_asli.endswith(clean_preview[-20:]):
            preview = f"{preview}..."
            
    else:
        # 2. Fallback: Jika mode pure KNN (tidak ada exact text match) atau text terlalu pendek
        preview = terjemahan_asli[:300].strip()
        if len(terjemahan_asli) > 300:
            preview += "..."

    return HaditsResult(
        nama_perawi=src.get("nama_perawi", ""),
        nomor_hadits=src.get("nomor_hadits", 0),
        referensi_lengkap=src.get("referensi_lengkap", ""),
        arab=src.get("arab", ""),
        terjemahan=terjemahan_asli,
        preview=preview,  # Masukkan preview ke response
        score=score,
    )


def _parse_suggestion(suggest_block: dict | None) -> str | None:
    if not suggest_block or "spell_check" not in suggest_block:
        return None
    
    spell_check = suggest_block["spell_check"]
    if not spell_check:
        return None
        
    has_suggestion = False
    words = []
    
    for item in spell_check:
        original_text = item.get("text", "")
        options = item.get("options", [])
        
        if options:
            best_option = options[0]["text"]
            words.append(best_option)
            has_suggestion = True
        else:
            words.append(original_text)
            
    if has_suggestion:
        return " ".join(words)
    return None


def _get_default_threshold(mode: str) -> float:
    return {
        "knn": 0.5,
        "bm25": 5.0,
        "hybrid": 1.0
    }.get(mode, 0.0)


def _clean_suggestion_fragment(fragment: str) -> str:
    text = fragment.replace("**", " ")
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    return text


def _normalize_suggestion_words(text: str) -> list[str]:
    words = []
    for raw_word in text.split():
        word = re.sub(r"(^[^\w]+|[^\w]+$)", "", raw_word, flags=re.UNICODE)
        if word:
            words.append(word)
    return words


def _find_query_start(words: list[str], query_terms: list[str]) -> int | None:
    if not words or not query_terms:
        return None

    normalized_words = [word.lower() for word in words]
    term_count = len(query_terms)

    for start in range(0, len(normalized_words) - term_count + 1):
        matches = True
        for offset, term in enumerate(query_terms):
            candidate = normalized_words[start + offset]
            is_last_term = offset == term_count - 1
            if is_last_term:
                if not candidate.startswith(term):
                    matches = False
                    break
            elif candidate != term:
                matches = False
                break
        if matches:
            return start

    return None


def _to_next_word_suggestion(text: str, query: str) -> str | None:
    words = _normalize_suggestion_words(text)
    query_terms = [term.lower() for term in _normalize_suggestion_words(query)]
    if not words or not query_terms:
        return None

    start = _find_query_start(words, query_terms)
    if start is None:
        return None

    suggestion_end = min(len(words), start + max(len(query_terms) + 3, 4))
    suggestion_words = words[start:suggestion_end]
    if not suggestion_words:
        return None

    return " ".join(suggestion_words).strip().lower()


def _extract_suggestion_from_hit(hit: dict, query: str) -> str | None:
    highlights = hit.get("highlight", {}).get("terjemahan", [])
    if highlights:
        suggestion = _to_next_word_suggestion(_clean_suggestion_fragment(highlights[0]), query)
        if suggestion:
            return suggestion

    text = hit.get("_source", {}).get("terjemahan", "")
    if not text:
        return None

    return _to_next_word_suggestion(_clean_suggestion_fragment(text), query)


def search_hadits(
    client: OpenSearch,
    query: str,
    page: int = 1,
    page_size: int = 10,
    mode: str = "knn",
    threshold: float | None = None,
) -> SearchResponse:
    embedding = get_model().encode(query).tolist()
    strategy = get_strategy(mode)
    body = strategy.build_query(query_text=query, embedding=embedding, page=page, page_size=page_size)

    body["suggest"] = {
        "text": query,
        "spell_check": {
            "term": {
                "field": "terjemahan",
                "suggest_mode": "missing"
            }
        }
    }

    actual_threshold = threshold if threshold is not None else _get_default_threshold(mode)
    if actual_threshold > 0.0:
        body["min_score"] = actual_threshold

    # TAMBAHKAN KONFIGURASI HIGHLIGHTING KE OPENSEARCH
    body["highlight"] = {
        "pre_tags": ["**"],  # Tag pembuka markdown bold
        "post_tags": ["**"], # Tag penutup markdown bold
        "fields": {
            "terjemahan": {
                "fragment_size": 300, # Batasi sekitar 160 karakter (Standar Google Snippet)
                "number_of_fragments": 1 # Ambil 1 kutipan terbaik saja
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    total = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]

    suggestion = _parse_suggestion(response.get("suggest"))

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=total, suggestion=suggestion, results=results)

def advanced_search_hadits(
    client: OpenSearch,
    query: str,
    page: int = 1,
    page_size: int = 10,
    nama_perawi: str | None = None,
    mode: str = "knn",
    threshold: float | None = None,
) -> SearchResponse:
    embedding = get_model().encode(query).tolist()
    strategy = get_strategy(mode)
    body = strategy.build_query(query_text=query, embedding=embedding, page=page, page_size=page_size)

    body["suggest"] = {
        "text": query,
        "spell_check": {
            "term": {
                "field": "terjemahan",
                "suggest_mode": "missing"
            }
        }
    }

    actual_threshold = threshold if threshold is not None else _get_default_threshold(mode)
    if actual_threshold > 0.0:
        body["min_score"] = actual_threshold

    # TAMBAHKAN KONFIGURASI HIGHLIGHTING KE OPENSEARCH
    body["highlight"] = {
        "pre_tags": ["**"],
        "post_tags": ["**"],
        "fields": {
            "terjemahan": {
                "fragment_size": 300,
                "number_of_fragments": 1
            }
        }
    }

    if nama_perawi:
        body["query"] = {
            "bool": {
                "must": body["query"],
                "filter": [{"term": {"nama_perawi": nama_perawi}}],
            }
        }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    total = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]

    suggestion = _parse_suggestion(response.get("suggest"))

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=total, suggestion=suggestion, results=results)


def get_suggestions(client: OpenSearch, query: str) -> list[str]:
    """Mengambil saran autocomplete (rekomendasi pencarian) dari OpenSearch."""
    body = {
        "size": 5,
        "query": {
            "multi_match": {
                "query": query,
                "type": "bool_prefix",
                "fields": [
                    "terjemahan.suggest",
                    "terjemahan.suggest._2gram",
                    "terjemahan.suggest._3gram"
                ]
            }
        },
        "highlight": {
            "pre_tags": ["**"],
            "post_tags": ["**"],
            "fields": {
                "terjemahan": {
                    "type": "unified",
                    "number_of_fragments": 1,
                    "fragment_size": 120,
                    "matched_fields": [
                        "terjemahan",
                        "terjemahan.suggest",
                        "terjemahan.suggest._2gram",
                        "terjemahan.suggest._3gram",
                    ],
                }
            },
        }
    }
    
    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    
    suggestions = []
    seen = set()
    
    for hit in hits:
        suggestion_phrase = _extract_suggestion_from_hit(hit, query)

        if suggestion_phrase and suggestion_phrase not in seen:
            suggestions.append(suggestion_phrase)
            seen.add(suggestion_phrase)
            
    return suggestions
