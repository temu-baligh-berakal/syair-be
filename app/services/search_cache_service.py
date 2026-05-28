import hashlib
import json
import logging
import os
import re
from typing import Any

from app.config import INDEX_NAME
from app.schemas.hadits_schema import SearchResponse

logger = logging.getLogger(__name__)

SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "1800"))
POPULAR_SEARCH_CACHE_TTL_SECONDS = int(os.getenv("POPULAR_SEARCH_CACHE_TTL_SECONDS", "21600"))
POPULAR_SEARCH_THRESHOLD = int(os.getenv("POPULAR_SEARCH_THRESHOLD", "5"))
POPULAR_SEARCH_WINDOW_SECONDS = 24 * 60 * 60
SEARCH_CACHE_VERSION = os.getenv("SEARCH_CACHE_VERSION", "v1")

_redis_client = None
_redis_checked = False


def normalize_cache_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _get_redis_client():
    global _redis_client, _redis_checked

    if _redis_checked:
        return _redis_client

    _redis_checked = True
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None

    try:
        from redis import Redis

        _redis_client = Redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
    except Exception as e:
        logger.warning(f"Redis search cache tidak tersedia: {str(e)}")
        _redis_client = None

    return _redis_client


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_search_cache_key(
    *,
    query: str,
    mode: str,
    page: int,
    page_size: int,
    nama_perawi: str | None,
    threshold: float | None,
    use_reranker: bool,
    reranker_id: str | None = None,
    reranker_candidate_limit: int | None = None,
) -> str:
    payload = {
        "index": INDEX_NAME,
        "version": SEARCH_CACHE_VERSION,
        "query": normalize_cache_query(query),
        "mode": mode,
        "page": page,
        "page_size": page_size,
        "nama_perawi": nama_perawi or "",
        "threshold": threshold,
        "use_reranker": use_reranker,
        "reranker_id": reranker_id or "",
        "reranker_candidate_limit": reranker_candidate_limit,
    }
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"syair:search_cache:{SEARCH_CACHE_VERSION}:{digest}"


def _popular_counter_key(query: str) -> str:
    digest = hashlib.sha256(normalize_cache_query(query).encode("utf-8")).hexdigest()
    return f"syair:search_popular:{SEARCH_CACHE_VERSION}:{digest}"


def record_search_query_and_get_popularity(query: str) -> bool:
    client = _get_redis_client()
    if client is None:
        return False

    try:
        key = _popular_counter_key(query)
        count = client.incr(key)
        if count == 1:
            client.expire(key, POPULAR_SEARCH_WINDOW_SECONDS)
        return int(count) >= POPULAR_SEARCH_THRESHOLD
    except Exception as e:
        logger.warning(f"Gagal menghitung popular query Redis: {str(e)}")
        return False


def get_cached_search_response(cache_key: str) -> SearchResponse | None:
    client = _get_redis_client()
    if client is None:
        return None

    try:
        cached = client.get(cache_key)
        if not cached:
            return None
        return SearchResponse.model_validate_json(cached)
    except Exception as e:
        logger.warning(f"Gagal membaca cache search Redis: {str(e)}")
        return None


def set_cached_search_response(
    cache_key: str,
    response: SearchResponse,
    *,
    is_popular: bool,
) -> None:
    client = _get_redis_client()
    if client is None:
        return

    ttl = POPULAR_SEARCH_CACHE_TTL_SECONDS if is_popular else SEARCH_CACHE_TTL_SECONDS
    try:
        client.setex(cache_key, ttl, response.model_dump_json())
    except Exception as e:
        logger.warning(f"Gagal menyimpan cache search Redis: {str(e)}")
