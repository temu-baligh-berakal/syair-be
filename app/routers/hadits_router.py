# app/routers/hadits_router.py
from fastapi import APIRouter, Depends, HTTPException
from opensearchpy import OpenSearch

from app.schemas.hadits_schema import SearchQuery, SearchResponse, SuggestionResponse, HaditsDetailResponse
from app.services.opensearch_client import get_opensearch_client
from app.services.hadits_service import search_hadits, advanced_search_hadits, get_suggestions, get_hadits_by_id, get_related_hadits
router = APIRouter(prefix="/hadits", tags=["Hadits"])


def get_client() -> OpenSearch:
    """Dependency injection untuk OpenSearch client."""
    return get_opensearch_client()


@router.post("/search", response_model=SearchResponse, summary="Pencarian Semantik Hadits")
def search(query: SearchQuery, client: OpenSearch = Depends(get_client)):
    """
    Cari hadits menggunakan pencarian semantik (KNN embedding).

    - **query**: kalimat pencarian dalam bahasa Indonesia
    - **page**: nomor halaman (default: 1)
    - **page_size**: jumlah hasil per halaman (default: 10)
    """
    try:
        return search_hadits(
            client=client, 
            query=query.query, 
            page=query.page,
            page_size=query.page_size,
            mode=query.mode,
            threshold=query.threshold,
            use_reranker=query.use_reranker
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghubungi OpenSearch: {str(e)}")


@router.post("/advanced-search", response_model=SearchResponse, summary="Pencarian Lanjutan Hadits")
def advanced_search(query: SearchQuery, client: OpenSearch = Depends(get_client)):
    # ... (existing code, unchanged for brevity)
    try:
        return advanced_search_hadits(
            client=client,
            query=query.query,
            page=query.page,
            page_size=query.page_size,
            nama_perawi=query.nama_perawi,
            mode=query.mode,
            threshold=query.threshold,
            use_reranker=query.use_reranker
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghubungi OpenSearch: {str(e)}")


@router.get("/{hadits_id}", response_model=HaditsDetailResponse, summary="Detil Hadits dan Rekomendasi")
def get_detail(hadits_id: str, client: OpenSearch = Depends(get_client)):
    """
    Ambil detil satu hadits beserta 10 hadits yang paling mirip secara semantik.
    - **hadits_id**: ID dokumen OpenSearch (didapatkan dari endpoint /search)
    """
    try:
        hadits = get_hadits_by_id(client, hadits_id)
        related = get_related_hadits(client, hadits_id)
        return HaditsDetailResponse(hadits=hadits, related_hadits=related)
    except Exception as e:
        if "404" in str(e):
            raise HTTPException(status_code=404, detail="Hadits tidak ditemukan")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggest", response_model=SuggestionResponse, summary="Autocomplete / Rekomendasi Pencarian")
def suggest(q: str, client: OpenSearch = Depends(get_client)):
    """
    Memberikan rekomendasi kata/frasa saat user mengetik (Google-like).
    - **q**: query parsial dari user
    """
    if len(q) < 2:
        return SuggestionResponse(query=q, suggestions=[])
    try:
        suggestions = get_suggestions(client, q)
        return SuggestionResponse(query=q, suggestions=suggestions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
