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
    source = hit.get("_source", {})
    return HaditsResult(
        nama_perawi=source.get("nama_perawi", ""),
        nomor_hadits=source.get("nomor_hadits", 0),
        referensi_lengkap=source.get("referensi_lengkap", ""),
        arab=source.get("arab", ""),
        terjemahan=source.get("terjemahan", ""),
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
