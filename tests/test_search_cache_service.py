from app.schemas.hadits_schema import SearchResponse
from app.services import search_cache_service as cache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.counts = {}
        self.expiries = {}
        self.last_ttl = None

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, seconds):
        self.expiries[key] = seconds

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.last_ttl = ttl
        self.values[key] = value


class ErrorRedis:
    def incr(self, key):
        raise RuntimeError("redis down")

    def get(self, key):
        raise RuntimeError("redis down")

    def setex(self, key, ttl, value):
        raise RuntimeError("redis down")


def test_popular_query_threshold(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_get_redis_client", lambda: fake)
    monkeypatch.setattr(cache, "POPULAR_SEARCH_THRESHOLD", 5)

    for _ in range(4):
        assert cache.record_search_query_and_get_popularity("Keutamaan ilmu") is False

    assert cache.record_search_query_and_get_popularity("Keutamaan ilmu") is True
    assert list(fake.expiries.values()) == [cache.POPULAR_SEARCH_WINDOW_SECONDS]


def test_cache_ttl_normal_dan_popular(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_get_redis_client", lambda: fake)
    monkeypatch.setattr(cache, "SEARCH_CACHE_TTL_SECONDS", 30)
    monkeypatch.setattr(cache, "POPULAR_SEARCH_CACHE_TTL_SECONDS", 360)

    response = SearchResponse(query="q", total=0, results=[])
    cache.set_cached_search_response("normal", response, is_popular=False)
    assert fake.last_ttl == 30

    cache.set_cached_search_response("popular", response, is_popular=True)
    assert fake.last_ttl == 360


def test_cache_response_roundtrip(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_get_redis_client", lambda: fake)

    response = SearchResponse(query="q", total=0, results=[])
    cache.set_cached_search_response("key", response, is_popular=False)

    cached = cache.get_cached_search_response("key")
    assert cached == response


def test_cache_key_membedakan_reranker_id():
    base_kwargs = {
        "query": "Keutamaan ilmu",
        "mode": "hybrid",
        "page": 1,
        "page_size": 10,
        "nama_perawi": None,
        "threshold": None,
        "use_reranker": True,
    }

    jina_key = cache.build_search_cache_key(**base_kwargs, reranker_id="jina:jina-reranker-v3")
    local_key = cache.build_search_cache_key(**base_kwargs, reranker_id="local:cross-encoder")

    assert jina_key != local_key


def test_cache_key_membedakan_limit_kandidat_reranker():
    base_kwargs = {
        "query": "Keutamaan ilmu",
        "mode": "hybrid",
        "page": 1,
        "page_size": 10,
        "nama_perawi": None,
        "threshold": None,
        "use_reranker": True,
        "reranker_id": "jina:jina-reranker-v3",
    }

    top_10_key = cache.build_search_cache_key(**base_kwargs, reranker_candidate_limit=10)
    top_100_key = cache.build_search_cache_key(**base_kwargs, reranker_candidate_limit=100)

    assert top_10_key != top_100_key


def test_redis_error_dianggap_cache_miss(monkeypatch):
    monkeypatch.setattr(cache, "_get_redis_client", lambda: ErrorRedis())

    assert cache.record_search_query_and_get_popularity("Keutamaan ilmu") is False
    assert cache.get_cached_search_response("key") is None
    cache.set_cached_search_response(
        "key",
        SearchResponse(query="q", total=0, results=[]),
        is_popular=False,
    )
