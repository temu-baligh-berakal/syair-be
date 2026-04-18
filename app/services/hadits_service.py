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
    """Konversi satu hit OpenSearch menjadi HaditsResult."""
    src = hit["_source"]
    return HaditsResult(
        nama_perawi=src.get("nama_perawi", ""),
        nomor_hadits=src.get("nomor_hadits", 0),
        referensi_lengkap=src.get("referensi_lengkap", ""),
        arab=src.get("arab", ""),
        terjemahan=src.get("terjemahan", ""),
        score=score,
    )


def _count_available_documents(
    client: OpenSearch,
    nama_perawi: str | None = None,
) -> int:
    body: dict | None = None

    if nama_perawi:
        body = {"query": {"term": {"nama_perawi": nama_perawi}}}

    response = client.count(index=INDEX_NAME, body=body)
    return int(response.get("count", 0))


def _resolve_effective_top_k(
    client: OpenSearch,
    requested_top_k: int,
    nama_perawi: str | None = None,
) -> int:
    available_docs = _count_available_documents(client=client, nama_perawi=nama_perawi)

    if available_docs <= 0:
        return 0

    return min(requested_top_k, available_docs)


def search_hadits(
    client: OpenSearch,
    query: str,
    top_k: int = 10,
    mode: str = "knn",
) -> SearchResponse:
    effective_top_k = _resolve_effective_top_k(client=client, requested_top_k=top_k)

    if effective_top_k == 0:
        return SearchResponse(query=query, total=0, results=[])

    embedding = get_model().encode(query).tolist()
    strategy = get_strategy(mode)
    body = strategy.build_query(query_text=query, embedding=embedding, top_k=effective_top_k)

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=len(results), results=results)


def advanced_search_hadits(
    client: OpenSearch,
    query: str,
    top_k: int = 10,
    nama_perawi: str | None = None,
    mode: str = "knn",
) -> SearchResponse:
    effective_top_k = _resolve_effective_top_k(
        client=client,
        requested_top_k=top_k,
        nama_perawi=nama_perawi,
    )

    if effective_top_k == 0:
        return SearchResponse(query=query, total=0, results=[])

    embedding = get_model().encode(query).tolist()
    strategy = get_strategy(mode)
    body = strategy.build_query(query_text=query, embedding=embedding, top_k=effective_top_k)

    if nama_perawi:
        body["query"] = {
            "bool": {
                "must": body["query"],
                "filter": [{"term": {"nama_perawi": nama_perawi}}],
            }
        }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=len(results), results=results)
