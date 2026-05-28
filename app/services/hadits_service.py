import json
import re
import logging
import os
import psycopg2
import urllib.error
import urllib.request
import time
from typing import Any

from opensearchpy import OpenSearch

from app.config import INDEX_NAME
from app.schemas.hadits_schema import HaditsResult, SearchResponse
from app.services.feedback_service import get_irrelevant_feedback_ids
from app.services.search_cache_service import (
    build_search_cache_key,
    get_cached_search_response,
    record_search_query_and_get_popularity,
    set_cached_search_response,
)
from app.services.strategies import get_strategy

import app.services.strategies.knn     # noqa: F401
import app.services.strategies.bm25    # noqa: F401
import app.services.strategies.hybrid  # noqa: F401

logger = logging.getLogger(__name__)

_model: Any | None = None
_cross_encoder: Any | None = None
LOCAL_RERANKER_MODEL_NAME = os.getenv(
    "LOCAL_RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L2-v2",
)
JINA_RERANKER_API_URL = os.getenv("JINA_RERANKER_API_URL", "https://api.jina.ai/v1/rerank")
JINA_RERANKER_MODEL = os.getenv("JINA_RERANKER_MODEL", "jina-reranker-v3")
JINA_RERANKER_TIMEOUT_SECONDS = float(os.getenv("JINA_RERANKER_TIMEOUT_SECONDS", "10"))
JINA_RERANKER_USER_AGENT = os.getenv("JINA_RERANKER_USER_AGENT", "SyairBackend/0.1")
EXTERNAL_RERANKER_RATE_LIMIT_COOLDOWN_SECONDS = float(
    os.getenv("EXTERNAL_RERANKER_RATE_LIMIT_COOLDOWN_SECONDS", "60")
)
LOW_CONFIDENCE_RERANKER_SCORE = float(os.getenv("LOW_CONFIDENCE_RERANKER_SCORE", "-1.5"))
RERANKER_CANDIDATE_SIZE = int(os.getenv("RERANKER_CANDIDATE_SIZE", "100"))
RERANKER_MAX_CANDIDATES = int(os.getenv("RERANKER_MAX_CANDIDATES", "100"))
RELATED_HADITS_LIMIT = 10
_external_reranker_cooldown_until: dict[str, float] = {}

_SEARCH_STOP_TERMS = {
    "ada", "agar", "akan", "apa", "atau", "bagaimana", "bagi", "cara",
    "dalam", "dan", "dari", "dengan", "di", "ini", "itu", "ke", "lebih",
    "mengenai", "pada", "tentang", "untuk", "yang",
}

_PROCEDURAL_QUERY_HINTS = {
    "cara", "bagaimana", "langkah", "tata", "prosedur", "urutan",
}

_PROCEDURAL_DOC_HINTS = {
    "cara", "langkah", "tata", "berwudhu", "wudhu", "mencuci", "membasuh",
    "mengguyur", "menyiram", "menuangkan", "mengusap", "memulai", "kemudian",
    "lalu", "setelah", "kepala", "rambut", "tangan", "kaki", "badan",
}

def get_model() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

def get_cross_encoder() -> Any:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(LOCAL_RERANKER_MODEL_NAME)
    return _cross_encoder


def _get_jina_api_key() -> str | None:
    return os.getenv("JINA_API_KEY") or os.getenv("JINA_AUTH_TOKEN")


def _get_reranker_provider(reranker_provider: str | None = None) -> str:
    provider = (reranker_provider or os.getenv("RERANKER_PROVIDER", "jina")).strip().lower()
    if provider == "voyage":
        logger.warning("Reranker provider 'voyage' sudah deprecated; memakai jina.")
        return "jina"
    if provider not in {"jina", "local"}:
        logger.warning(f"Reranker provider '{provider}' tidak dikenal; fallback ke local.")
        return "local"
    return provider


def _should_use_jina_reranker(reranker_provider: str | None = None) -> bool:
    return _get_reranker_provider(reranker_provider) == "jina" and bool(_get_jina_api_key())


def _get_reranker_cache_id(reranker_provider: str | None = None) -> str:
    provider = _get_reranker_provider(reranker_provider)
    if provider == "jina" and _get_jina_api_key():
        return f"jina:{os.getenv('JINA_RERANKER_MODEL', JINA_RERANKER_MODEL)}"
    return f"local:{LOCAL_RERANKER_MODEL_NAME}"


def _external_reranker_cooldown_remaining(provider: str) -> float:
    return max(0.0, _external_reranker_cooldown_until.get(provider, 0.0) - time.time())


def _mark_external_reranker_rate_limited(provider: str) -> None:
    cooldown_seconds = max(0.0, EXTERNAL_RERANKER_RATE_LIMIT_COOLDOWN_SECONDS)
    if cooldown_seconds <= 0:
        return

    _external_reranker_cooldown_until[provider] = time.time() + cooldown_seconds
    logger.warning(
        f"RERANKER_PROVIDER={provider} kena rate limit; skip provider ini selama {cooldown_seconds:.0f} detik dan fallback ke local."
    )


def preload_reranker() -> None:
    provider = _get_reranker_provider()
    if provider == "jina" and _get_jina_api_key():
        logger.info(
            f"RERANKER_PROVIDER=jina aktif; menggunakan Jina model={os.getenv('JINA_RERANKER_MODEL', JINA_RERANKER_MODEL)}. "
            "Local CrossEncoder tidak dipreload."
        )
        return
    if provider == "jina" and not _get_jina_api_key():
        logger.warning("JINA_API_KEY belum diset; fallback ke local CrossEncoder.")
    logger.info(f"RERANKER_PROVIDER={provider}; menggunakan local CrossEncoder model={LOCAL_RERANKER_MODEL_NAME}.")
    get_cross_encoder()

def _parse_hit(hit: dict, score: float) -> HaditsResult:
    """Konversi satu hit OpenSearch menjadi HaditsResult dan ekstrak Highlight."""
    src = hit.get("_source", {})
    hadits_id = hit.get("_id") or src.get("id") or f"{src.get('nama_perawi', '')}-{src.get('nomor_hadits', 0)}"
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
        id=str(hadits_id),
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


def _normalize_query_terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"\w+", text, flags=re.UNICODE)]


def _content_query_terms(query: str) -> list[str]:
    return [
        term
        for term in _normalize_query_terms(query)
        if len(term) >= 3 and term not in _SEARCH_STOP_TERMS
    ]


def _has_query_term_overlap(query: str, result: HaditsResult) -> bool:
    query_terms = _content_query_terms(query)
    if not query_terms:
        return False

    searchable_terms = set(
        _normalize_query_terms(
            f"{result.referensi_lengkap} {result.preview} {result.terjemahan}"
        )
    )
    return any(term in searchable_terms for term in query_terms)


def _filter_low_confidence_reranked_results(
    query: str,
    results: list[HaditsResult],
) -> list[HaditsResult]:
    return [
        result
        for result in results
        if result.score >= LOW_CONFIDENCE_RERANKER_SCORE
        or _has_query_term_overlap(query, result)
    ]


def _filter_irrelevant_feedback(
    query: str,
    results: list[HaditsResult],
) -> tuple[list[HaditsResult], set[str]]:
    irrelevant_ids = get_irrelevant_feedback_ids(query)
    if not irrelevant_ids:
        return results, set()

    return [
        result
        for result in results
        if result.id not in irrelevant_ids
    ], irrelevant_ids


def _apply_irrelevant_feedback(
    query: str,
    response: SearchResponse,
) -> SearchResponse:
    filtered_results, irrelevant_feedback_ids = _filter_irrelevant_feedback(
        query,
        response.results,
    )
    removed_count = len(response.results) - len(filtered_results)
    if response.total <= len(response.results):
        total = max(0, response.total - removed_count)
    else:
        total = max(0, response.total - len(irrelevant_feedback_ids))
    return SearchResponse(
        query=response.query,
        total=total,
        suggestion=response.suggestion,
        results=filtered_results,
    )


def _get_reranker_fetch_size() -> int:
    return max(1, min(RERANKER_CANDIDATE_SIZE, RERANKER_MAX_CANDIDATES))


def _paginate_results(
    results: list[HaditsResult],
    page: int,
    page_size: int,
) -> list[HaditsResult]:
    start = (page - 1) * page_size
    end = start + page_size
    return results[start:end]


def _reranker_document_text(result: HaditsResult) -> str:
    return f"{result.referensi_lengkap}\nTerjemahan: {result.terjemahan}"


def _rerank_with_jina(
    query: str,
    results: list[HaditsResult],
    context: str,
) -> list[HaditsResult]:
    api_key = _get_jina_api_key()
    if not api_key:
        raise RuntimeError("JINA_API_KEY belum diset")

    model = os.getenv("JINA_RERANKER_MODEL", JINA_RERANKER_MODEL)
    documents = [_reranker_document_text(result) for result in results]
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
        "return_documents": False,
    }
    request = urllib.request.Request(
        os.getenv("JINA_RERANKER_API_URL", JINA_RERANKER_API_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": os.getenv("JINA_RERANKER_USER_AGENT", JINA_RERANKER_USER_AGENT),
        },
        method="POST",
    )

    logger.info(f"=== Top 10 BEFORE Jina Reranking ({context}: '{query}') ===")
    for i, result in enumerate(results[:10]):
        logger.info(
            f"{i+1}. [Score: {result.score:.4f}] "
            f"{result.nama_perawi} no. {result.nomor_hadits}"
        )

    try:
        with urllib.request.urlopen(request, timeout=JINA_RERANKER_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        if e.code == 429:
            _mark_external_reranker_rate_limited("jina")
        raise RuntimeError(f"Jina reranker HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Jina reranker request gagal: {e.reason}") from e

    data = json.loads(response_body)
    reranked: list[HaditsResult] = []
    seen_indices: set[int] = set()
    for item in data.get("results", []):
        index = item.get("index")
        if index is None:
            continue
        index = int(index)
        if index < 0 or index >= len(results):
            continue

        relevance_score = item.get("relevance_score", item.get("score"))
        if relevance_score is None:
            continue

        result = results[index]
        result.score = float(relevance_score)
        reranked.append(result)
        seen_indices.add(index)

    if not reranked:
        raise RuntimeError("Respons Jina reranker tidak berisi hasil valid")

    for index, result in enumerate(results):
        if index not in seen_indices:
            reranked.append(result)

    logger.info(f"=== Top 10 AFTER Jina Reranking ({context}: '{query}') ===")
    for i, result in enumerate(reranked[:10]):
        logger.info(
            f"{i+1}. [Score: {result.score:.4f}] "
            f"{result.nama_perawi} no. {result.nomor_hadits}"
        )

    return reranked


def _rerank_with_cross_encoder(
    query: str,
    results: list[HaditsResult],
    context: str,
) -> list[HaditsResult]:
    logger.info(f"=== Top 10 BEFORE Reranking ({context}: '{query}') ===")
    for i, result in enumerate(results[:10]):
        logger.info(
            f"{i+1}. [Score: {result.score:.4f}] "
            f"{result.nama_perawi} no. {result.nomor_hadits}"
        )

    ce_model = get_cross_encoder()
    pairs = [[query, result.terjemahan] for result in results]
    ce_scores = ce_model.predict(pairs)

    for result, score in zip(results, ce_scores):
        result.score = float(score)

    reranked = sorted(results, key=lambda x: x.score, reverse=True)

    logger.info(f"=== Top 10 AFTER Reranking ({context}: '{query}') ===")
    for i, result in enumerate(reranked[:10]):
        logger.info(
            f"{i+1}. [Score: {result.score:.4f}] "
            f"{result.nama_perawi} no. {result.nomor_hadits}"
        )

    return reranked


def _rerank_results(
    query: str,
    results: list[HaditsResult],
    context: str,
    reranker_provider: str | None = None,
) -> list[HaditsResult]:
    provider = _get_reranker_provider(reranker_provider)
    logger.info(f"Reranker switch check: requested={reranker_provider or 'env'}, resolved={provider}, candidates={len(results)}")

    if provider == "jina" and _get_jina_api_key():
        cooldown_remaining = _external_reranker_cooldown_remaining("jina")
        if cooldown_remaining > 0:
            logger.warning(
                f"RERANKER_PROVIDER=jina masih cooldown {cooldown_remaining:.0f} detik; fallback ke local CrossEncoder."
            )
        else:
            try:
                return _rerank_with_jina(query, results, context)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    _mark_external_reranker_rate_limited("jina")
                logger.warning(f"Jina reranker gagal, fallback ke local CrossEncoder: {str(e)}")
            except Exception as e:
                logger.warning(f"Jina reranker gagal, fallback ke local CrossEncoder: {str(e)}")

    if provider == "jina" and not _get_jina_api_key():
        logger.warning("RERANKER_PROVIDER=jina dipilih tapi JINA_API_KEY kosong; fallback ke local CrossEncoder.")
    logger.info(f"RERANKER_PROVIDER=local berjalan ({context}); model={LOCAL_RERANKER_MODEL_NAME}, candidates={len(results)}")
    return _rerank_with_cross_encoder(query, results, context)


def _is_procedural_query(query: str) -> bool:
    terms = _normalize_query_terms(query)
    return any(term in _PROCEDURAL_QUERY_HINTS for term in terms)


def _procedural_topic_terms(query: str) -> list[str]:
    stop_terms = _PROCEDURAL_QUERY_HINTS | {
        "itu", "ini", "yang", "dan", "atau", "apa", "agar", "untuk",
    }
    return [term for term in _normalize_query_terms(query) if term not in stop_terms]


def _rerank_for_procedural_query(query: str, results: list[HaditsResult]) -> list[HaditsResult]:
    if not _is_procedural_query(query) or len(results) < 2:
        return results

    topic_terms = _procedural_topic_terms(query)

    def ranking_key(result: HaditsResult) -> tuple[int, int, int, float]:
        searchable_text = f"{result.preview} {result.terjemahan}".lower()
        topic_matches = sum(1 for term in topic_terms if term in searchable_text)
        doc_hints = sum(1 for hint in _PROCEDURAL_DOC_HINTS if hint in searchable_text)
        has_sequence = int(any(token in searchable_text for token in ("kemudian", "lalu", "setelah")))
        return (
            int(topic_matches > 0 and doc_hints > 0),
            has_sequence,
            doc_hints,
            result.score,
        )

    return sorted(results, key=ranking_key, reverse=True)


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


def get_hadits_by_id(client: OpenSearch, hadits_id: str) -> HaditsResult:
    """Ambil satu hadits berdasarkan ID OpenSearch."""
    hit = client.get(index=INDEX_NAME, id=hadits_id)
    return _parse_hit(hit, 1.0)


def _get_precomputed_related_hadits(
    client: OpenSearch,
    hadits_id: str,
    limit: int,
) -> list[HaditsResult]:
    """Ambil related hadits dari tabel precomputed."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.info("DATABASE_URL tidak ditemukan; Hadits Serupa kosong karena semantic fallback dinonaktifkan.")
        return []

    conn = None
    try:
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT target_id, score
                FROM hadits_similarity
                WHERE source_id = %s
                ORDER BY score DESC
                LIMIT %s
                """,
                (hadits_id, limit),
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.warning(f"Gagal membaca tabel hadits_similarity: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

    try:
        if not rows:
            return []

        target_ids = [r[0] for r in rows]
        scores = {r[0]: r[1] for r in rows}
        res = client.mget(index=INDEX_NAME, body={"ids": target_ids})

        return [
            _parse_hit(hit, scores[hit["_id"]])
            for hit in res["docs"]
            if hit.get("found")
        ]
    except Exception as e:
        logger.warning(f"Gagal mengambil konten related precomputed: {str(e)}")
        return []


def get_related_hadits(client: OpenSearch, hadits_id: str) -> list[HaditsResult]:
    """Ambil daftar hadits mirip dari tabel precomputed database."""
    related = _get_precomputed_related_hadits(client, hadits_id, RELATED_HADITS_LIMIT)
    filtered, _ = _filter_irrelevant_feedback(f"related:{hadits_id}", related)
    return filtered



def search_hadits(
    client: OpenSearch,
    query: str,
    page: int = 1,
    page_size: int = 10,
    mode: str = "knn",
    threshold: float | None = None,
    use_reranker: bool = True,
    reranker_provider: str | None = None,
) -> SearchResponse:
    resolved_reranker_provider = _get_reranker_provider(reranker_provider)
    is_popular = record_search_query_and_get_popularity(query)
    cache_key = build_search_cache_key(
        query=query,
        mode=mode,
        page=page,
        page_size=page_size,
        nama_perawi=None,
        threshold=threshold,
        use_reranker=use_reranker,
        reranker_id=_get_reranker_cache_id(resolved_reranker_provider) if use_reranker else None,
        reranker_candidate_limit=_get_reranker_fetch_size() if use_reranker else None,
    )
    logger.info(
        f"Search reranker setting: enabled={use_reranker}, requested={reranker_provider or 'env'}, "
        f"resolved={resolved_reranker_provider}, cache_id={_get_reranker_cache_id(resolved_reranker_provider) if use_reranker else 'none'}"
    )
    cached_response = get_cached_search_response(cache_key)
    if cached_response is not None:
        if is_popular:
            set_cached_search_response(cache_key, cached_response, is_popular=True)
        return _apply_irrelevant_feedback(query, cached_response)

    candidate_response = _search_hadits_candidates(
        client=client,
        query=query,
        page=page,
        page_size=page_size,
        mode=mode,
        threshold=threshold,
        use_reranker=use_reranker,
        reranker_provider=resolved_reranker_provider,
    )
    set_cached_search_response(cache_key, candidate_response, is_popular=is_popular)
    return _apply_irrelevant_feedback(query, candidate_response)


def _search_hadits_candidates(
    client: OpenSearch,
    query: str,
    page: int = 1,
    page_size: int = 10,
    mode: str = "knn",
    threshold: float | None = None,
    use_reranker: bool = True,
    reranker_provider: str | None = None,
) -> SearchResponse:
    # Saat reranker aktif, ambil top-N kandidat dari awal, rerank semuanya,
    # baru pagination setelah skor reranker final.
    fetch_page = 1 if use_reranker else page
    fetch_size = _get_reranker_fetch_size() if use_reranker else page_size
    
    embedding = get_model().encode(query).tolist()
    strategy = get_strategy(mode)
    body = strategy.build_query(query_text=query, embedding=embedding, page=fetch_page, page_size=fetch_size)

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
    
    # --- RERANKING ---
    if use_reranker and results:
        results = _rerank_results(query, results, "search", reranker_provider)
        results = _filter_low_confidence_reranked_results(query, results)

    results = _rerank_for_procedural_query(query, results)

    if use_reranker:
        total = len(results)
        paginated_results = _paginate_results(results, page, page_size)
    else:
        paginated_results = results
    
    return SearchResponse(query=query, total=total, suggestion=suggestion, results=paginated_results)

def advanced_search_hadits(
    client: OpenSearch,
    query: str,
    page: int = 1,
    page_size: int = 10,
    nama_perawi: str | None = None,
    mode: str = "knn",
    threshold: float | None = None,
    use_reranker: bool = True,
    reranker_provider: str | None = None,
) -> SearchResponse:
    resolved_reranker_provider = _get_reranker_provider(reranker_provider)
    is_popular = record_search_query_and_get_popularity(query)
    cache_key = build_search_cache_key(
        query=query,
        mode=mode,
        page=page,
        page_size=page_size,
        nama_perawi=nama_perawi,
        threshold=threshold,
        use_reranker=use_reranker,
        reranker_id=_get_reranker_cache_id(resolved_reranker_provider) if use_reranker else None,
        reranker_candidate_limit=_get_reranker_fetch_size() if use_reranker else None,
    )
    logger.info(
        f"Advanced search reranker setting: enabled={use_reranker}, requested={reranker_provider or 'env'}, "
        f"resolved={resolved_reranker_provider}, cache_id={_get_reranker_cache_id(resolved_reranker_provider) if use_reranker else 'none'}"
    )
    cached_response = get_cached_search_response(cache_key)
    if cached_response is not None:
        if is_popular:
            set_cached_search_response(cache_key, cached_response, is_popular=True)
        return _apply_irrelevant_feedback(query, cached_response)

    candidate_response = _advanced_search_hadits_candidates(
        client=client,
        query=query,
        page=page,
        page_size=page_size,
        nama_perawi=nama_perawi,
        mode=mode,
        threshold=threshold,
        use_reranker=use_reranker,
        reranker_provider=resolved_reranker_provider,
    )
    set_cached_search_response(cache_key, candidate_response, is_popular=is_popular)
    return _apply_irrelevant_feedback(query, candidate_response)


def _advanced_search_hadits_candidates(
    client: OpenSearch,
    query: str,
    page: int = 1,
    page_size: int = 10,
    nama_perawi: str | None = None,
    mode: str = "knn",
    threshold: float | None = None,
    use_reranker: bool = True,
    reranker_provider: str | None = None,
) -> SearchResponse:
    fetch_page = 1 if use_reranker else page
    fetch_size = _get_reranker_fetch_size() if use_reranker else page_size
    
    embedding = get_model().encode(query).tolist()
    strategy = get_strategy(mode)
    body = strategy.build_query(query_text=query, embedding=embedding, page=fetch_page, page_size=fetch_size)

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
    
    # --- RERANKING ---
    if use_reranker and results:
        results = _rerank_results(query, results, "advanced search", reranker_provider)
        results = _filter_low_confidence_reranked_results(query, results)
        
    results = _rerank_for_procedural_query(query, results)

    if use_reranker:
        total = len(results)
        paginated_results = _paginate_results(results, page, page_size)
    else:
        paginated_results = results

    return SearchResponse(query=query, total=total, suggestion=suggestion, results=paginated_results)


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
