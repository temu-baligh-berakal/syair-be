import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.routers.hadits_router import get_client
from app.services.hadits_service import (
    _parse_hit,
    search_hadits,
    advanced_search_hadits,
)

FAKE_EMBEDDING = np.zeros(384)

FAKE_HIT = {
    "_score": 0.92,
    "_source": {
        "nama_perawi": "Bukhari",
        "nomor_hadits": 1,
        "referensi_lengkap": "Hadits Bukhari Nomor 1",
        "arab": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
        "terjemahan": "Sesungguhnya setiap amalan tergantung pada niatnya.",
    },
}

def make_opensearch_response(hits: list[dict]) -> dict:
    """Buat response OpenSearch palsu."""
    return {"hits": {"hits": hits}}


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer agar tidak download model."""
    with patch("app.services.hadits_service.get_model") as m:
        model = MagicMock()
        model.encode.return_value = FAKE_EMBEDDING
        m.return_value = model
        yield model


@pytest.fixture
def mock_client():
    """Mock OpenSearch client."""
    client = MagicMock()
    client.search.return_value = make_opensearch_response([FAKE_HIT])
    return client


@pytest.fixture
def test_client(mock_client):
    """FastAPI TestClient dengan dependency get_client di-override."""
    app.dependency_overrides[get_client] = lambda: mock_client
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestParseHit:

    def test_parse_hit_normal(self):
        result = _parse_hit(FAKE_HIT, score=0.92)
        assert result.nama_perawi == "Bukhari"
        assert result.nomor_hadits == 1
        assert result.score == 0.92
        assert result.arab == "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ"

    def test_parse_hit_field_kosong(self):
        """Field yang tidak ada di _source diisi nilai default."""
        hit = {"_score": 0.5, "_source": {}}
        result = _parse_hit(hit, score=0.5)
        assert result.nama_perawi == ""
        assert result.nomor_hadits == 0
        assert result.terjemahan == ""


class TestSearchHaditsService:

    def test_memanggil_model_encode(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat ibadah", top_k=5)
        mock_model.encode.assert_called_once_with("niat ibadah")

    def test_memanggil_opensearch_search(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat ibadah", top_k=5)
        mock_client.search.assert_called_once()

    def test_query_knn_menggunakan_top_k(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", top_k=7)
        call_body = mock_client.search.call_args.kwargs["body"]
        assert call_body["size"] == 7
        assert call_body["query"]["knn"]["embedding"]["k"] == 7

    def test_return_search_response(self, mock_model, mock_client):
        resp = search_hadits(client=mock_client, query="niat", top_k=5)
        assert resp.query == "niat"
        assert resp.total == 1
        assert resp.results[0].nama_perawi == "Bukhari"

    def test_hasil_kosong(self, mock_model, mock_client):
        mock_client.search.return_value = make_opensearch_response([])
        resp = search_hadits(client=mock_client, query="tidak ada", top_k=5)
        assert resp.total == 0
        assert resp.results == []


class TestAdvancedSearchHaditsService:

    def test_tanpa_filter_pakai_knn_biasa(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="shalat", top_k=5, nama_perawi=None)
        body = mock_client.search.call_args.kwargs["body"]
        assert "knn" in body["query"]

    def test_dengan_filter_perawi_pakai_bool(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="shalat", top_k=5, nama_perawi="Bukhari")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]
        assert body["query"]["bool"]["filter"][0]["term"]["nama_perawi"] == "Bukhari"

    def test_return_search_response(self, mock_model, mock_client):
        resp = advanced_search_hadits(client=mock_client, query="niat", top_k=5, nama_perawi="Bukhari")
        assert resp.query == "niat"
        assert resp.total == 1

    def test_nama_perawi_kosong_tidak_difilter(self, mock_model, mock_client):
        """nama_perawi = None → tidak ada filter bool."""
        advanced_search_hadits(client=mock_client, query="zakat", top_k=5, nama_perawi=None)
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" not in body["query"]


class TestSearchRouter:

    def test_search_berhasil(self, mock_client, test_client):
        with patch("app.routers.hadits_router.search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse, HaditsResult
            mock_svc.return_value = SearchResponse(
                query="niat",
                total=1,
                results=[HaditsResult(
                    nama_perawi="Bukhari", nomor_hadits=1,
                    referensi_lengkap="Hadits Bukhari Nomor 1",
                    arab="...", terjemahan="...", score=0.9,
                )],
            )
            resp = test_client.post("/hadits/search", json={"query": "niat ibadah"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["nama_perawi"] == "Bukhari"

    def test_search_query_terlalu_pendek(self, test_client):
        resp = test_client.post("/hadits/search", json={"query": "ab"})
        assert resp.status_code == 422

    def test_search_tanpa_body(self, test_client):
        resp = test_client.post("/hadits/search")
        assert resp.status_code == 422

    def test_search_error_opensearch_return_500(self, mock_client, test_client):
        with patch("app.routers.hadits_router.search_hadits", side_effect=Exception("Connection refused")):
            resp = test_client.post("/hadits/search", json={"query": "niat ibadah"})
        assert resp.status_code == 500
        assert "Gagal menghubungi OpenSearch" in resp.json()["detail"]


class TestAdvancedSearchRouter:

    def test_advanced_search_tanpa_filter(self, mock_client, test_client):
        with patch("app.routers.hadits_router.advanced_search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse
            mock_svc.return_value = SearchResponse(query="shalat", total=0, results=[])
            resp = test_client.post("/hadits/advanced-search", json={"query": "shalat malam"})

        assert resp.status_code == 200

    def test_advanced_search_dengan_filter_perawi(self, mock_client, test_client):
        with patch("app.routers.hadits_router.advanced_search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse
            mock_svc.return_value = SearchResponse(query="puasa", total=0, results=[])
            resp = test_client.post(
                "/hadits/advanced-search",
                json={"query": "keutamaan puasa", "nama_perawi": "Muslim"},
            )
            _, kwargs = mock_svc.call_args
            assert kwargs.get("nama_perawi") == "Muslim"

        assert resp.status_code == 200

    def test_advanced_search_top_k_dikirim(self, mock_client, test_client):
        with patch("app.routers.hadits_router.advanced_search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse
            mock_svc.return_value = SearchResponse(query="zakat", total=0, results=[])
            test_client.post("/hadits/advanced-search", json={"query": "zakat fitrah", "top_k": 3})
            _, kwargs = mock_svc.call_args
            assert kwargs.get("top_k") == 3

    def test_advanced_search_error_return_500(self, mock_client, test_client):
        with patch("app.routers.hadits_router.advanced_search_hadits", side_effect=Exception("timeout")):
            resp = test_client.post("/hadits/advanced-search", json={"query": "shalat malam"})
        assert resp.status_code == 500
