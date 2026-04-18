# app/services/hadits_service.py
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

from app.config import INDEX_NAME
from app.schemas.hadits_schema import HaditsResult, SearchResponse

# Load model sekali saat startup (bukan per-request)
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


def search_hadits(
    client: OpenSearch,
    query: str,
    top_k: int = 10,
) -> SearchResponse:
    """
    Pencarian semantik menggunakan KNN embedding.
    Cocok untuk query bahasa alami seperti "niat dalam beribadah".
    """
    embedding = get_model().encode(query).tolist()

    body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": top_k,
                }
            }
        },
        "_source": ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"],
    }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=len(results), results=results)

def advanced_search_hadits(
    client: OpenSearch,
    query: str,
    top_k: int = 10,
    nama_perawi: str | None = None,
) -> SearchResponse:
    """
    Pencarian semantik dengan filter tambahan.
    - nama_perawi: filter exact match pada perawi (misal "Bukhari")
    Menggunakan post_filter agar KNN tetap berjalan lalu hasil difilter.
    """
    embedding = get_model().encode(query).tolist()

    knn_query: dict = {
        "knn": {
            "embedding": {
                "vector": embedding,
                "k": top_k,
            }
        }
    }

    filters = []
    if nama_perawi:
        filters.append({"term": {"nama_perawi": nama_perawi}})

    if filters:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": knn_query,
                    "filter": filters,
                }
            },
            "_source": ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"],
        }
    else:
        body = {
            "size": top_k,
            "query": knn_query,
            "_source": ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"],
        }

    response = client.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]

    results = [_parse_hit(h, h["_score"]) for h in hits]
    return SearchResponse(query=query, total=len(results), results=results)
