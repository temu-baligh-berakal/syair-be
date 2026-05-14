# app/routers/hadits_router.py
from fastapi import APIRouter, Depends, HTTPException
from opensearchpy import OpenSearch

from app.schemas.hadits_schema import SearchQuery, SearchResponse
from app.services.opensearch_client import get_opensearch_client
from app.services.hadits_service import search_hadits, advanced_search_hadits

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
            threshold=query.threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghubungi OpenSearch: {str(e)}")


@router.post("/advanced-search", response_model=SearchResponse, summary="Pencarian Lanjutan Hadits")
def advanced_search(query: SearchQuery, client: OpenSearch = Depends(get_client)):
    """
    Cari hadits dengan filter tambahan di atas pencarian semantik.

    - **query**: kalimat pencarian dalam bahasa Indonesia
    - **page**: nomor halaman (default: 1)
    - **page_size**: jumlah hasil per halaman (default: 10)
    - **nama_perawi**: filter perawi, misal `Bukhari`, `Muslim`, `Tirmidzi`
    """
    try:
        return advanced_search_hadits(
            client=client,
            query=query.query,
            page=query.page,
            page_size=query.page_size,
            nama_perawi=query.nama_perawi,
            mode=query.mode,
            threshold=query.threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menghubungi OpenSearch: {str(e)}")
