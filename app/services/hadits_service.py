from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

from app.config import INDEX_NAME
from app.schemas.hadits_schema import HaditsResult, SearchResponse, ByRawiResponse
from app.services.strategies import get_strategy

import app.services.strategies.knn  # noqa: F401
import app.services.strategies.bm25  # noqa: F401
import app.services.strategies.hybrid  # noqa: F401

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def _parse_hit(hit: dict, score: float) -> HaditsResult:
    """Konversi satu hit OpenSearch menjadi HaditsResult dan ekstrak Highlight."""
    src = hit["_source"]
    terjemahan_asli = src.get("terjemahan", "")

    # 1. Coba ambil highlight dari OpenSearch (Akan ada jika mode BM25 / Hybrid)
    highlights = hit.get("highlight", {}).get("terjemahan", [])

    if highlights:
        preview_raw = highlights[0]
        clean_preview = preview_raw.replace("**", "")

        preview = preview_raw

        # Tambahkan elipsis (...) di awal jika kutipan tidak dimulai dari awal kalimat
        if len(clean_preview) > 20 and not terjemahan_asli.startswith(
            clean_preview[:20]
        ):
            preview = f"...{preview}"

        # Tambahkan elipsis (...) di akhir jika kutipan terpotong sebelum akhir kalimat
        if len(clean_preview) > 20 and not terjemahan_asli.endswith(
            clean_preview[-20:]
        ):
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
    body = strategy.build_query(
        query_text=query, embedding=embedding, top_k=effective_top_k
    )

    # TAMBAHKAN KONFIGURASI HIGHLIGHTING KE OPENSEARCH
    body["highlight"] = {
        "pre_tags": ["**"],  # Tag pembuka markdown bold
        "post_tags": ["**"],  # Tag penutup markdown bold
        "fields": {
            "terjemahan": {
                "fragment_size": 300,  # Batasi sekitar 160 karakter (Standar Google Snippet)
                "number_of_fragments": 1,  # Ambil 1 kutipan terbaik saja
            }
        },
    }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=len(results), results=results)


def get_hadits_by_rawi(
    client: OpenSearch,
    rawi: str,
    page: int = 1,
    page_size: int = 10,
) -> ByRawiResponse:
    total = _count_available_documents(client=client, nama_perawi=rawi)

    if total == 0:
        return ByRawiResponse(
            rawi=rawi, total=0, page=page, page_size=page_size, results=[]
        )

    from_param = (page - 1) * page_size

    body = {
        "query": {"term": {"nama_perawi": rawi}},
        "sort": [{"nomor_hadits": "asc"}],
        "from": from_param,
        "size": page_size,
    }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]

    results = [_parse_hit(h, h.get("_score") or 0.0) for h in hits]
    return ByRawiResponse(
        rawi=rawi, total=total, page=page, page_size=page_size, results=results
    )


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
    body = strategy.build_query(
        query_text=query, embedding=embedding, top_k=effective_top_k
    )

    # TAMBAHKAN KONFIGURASI HIGHLIGHTING KE OPENSEARCH
    body["highlight"] = {
        "pre_tags": ["**"],
        "post_tags": ["**"],
        "fields": {"terjemahan": {"fragment_size": 300, "number_of_fragments": 1}},
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

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=len(results), results=results)
