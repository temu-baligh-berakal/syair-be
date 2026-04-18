import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.routers.hadits_router import get_client
from app.services.hadits_service import _parse_hit, search_hadits, advanced_search_hadits
from app.services.strategies import get_strategy, get_available_modes

# Import strategies agar terdaftar
import app.services.strategies.knn     # noqa: F401
import app.services.strategies.bm25    # noqa: F401
import app.services.strategies.hybrid  # noqa: F401


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
    return {"hits": {"hits": hits}}


@pytest.fixture
def mock_model():
    with patch("app.services.hadits_service.get_model") as m:
        model = MagicMock()
        model.encode.return_value = FAKE_EMBEDDING
        m.return_value = model
        yield model


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.search.return_value = make_opensearch_response([FAKE_HIT])
    client.count.return_value = {"count": 1}
    return client


@pytest.fixture
def test_client(mock_client):
    fastapi_app.dependency_overrides[get_client] = lambda: mock_client
    yield TestClient(fastapi_app)
    fastapi_app.dependency_overrides.clear()


class TestStrategyRegistry:

    def test_semua_mode_terdaftar(self):
        modes = get_available_modes()
        assert "knn" in modes
        assert "bm25" in modes
        assert "hybrid" in modes

    def test_get_strategy_valid(self):
        for mode in get_available_modes():
            strategy = get_strategy(mode)
            assert hasattr(strategy, "build_query")

    def test_get_strategy_tidak_dikenal(self):
        with pytest.raises(ValueError, match="tidak dikenal"):
            get_strategy("tidak_ada")

    def test_setiap_strategy_return_dict_dengan_query(self):
        embedding = [0.0] * 384
        for mode in get_available_modes():
            strategy = get_strategy(mode)
            body = strategy.build_query(query_text="test", embedding=embedding, top_k=5)
            assert "query" in body
            assert "size" in body
            assert body["size"] == 5


class TestParseHit:

    def test_parse_hit_normal(self):
        result = _parse_hit(FAKE_HIT, score=0.92)
        assert result.nama_perawi == "Bukhari"
        assert result.nomor_hadits == 1
        assert result.score == 0.92

    def test_parse_hit_field_kosong(self):
        hit = {"_score": 0.5, "_source": {}}
        result = _parse_hit(hit, score=0.5)
        assert result.nama_perawi == ""
        assert result.nomor_hadits == 0


class TestSearchHaditsService:

    def test_memanggil_model_encode(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat ibadah", top_k=5)
        mock_model.encode.assert_called_once_with("niat ibadah")

    def test_memanggil_opensearch_search(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat ibadah", top_k=5)
        mock_client.search.assert_called_once()

    def test_top_k_dibatasi_jumlah_dokumen_tersedia(self, mock_model, mock_client):
        mock_client.count.return_value = {"count": 3}

        search_hadits(client=mock_client, query="niat ibadah", top_k=50)

        body = mock_client.search.call_args.kwargs["body"]
        assert body["size"] == 3

        if "knn" in body["query"]:
            assert body["query"]["knn"]["embedding"]["k"] == 3

    def test_jika_dokumen_tidak_ada_return_kosong_tanpa_search(self, mock_model, mock_client):
        mock_client.count.return_value = {"count": 0}

        resp = search_hadits(client=mock_client, query="tidak ada", top_k=50)

        assert resp.total == 0
        mock_client.search.assert_not_called()

    def test_return_search_response(self, mock_model, mock_client):
        resp = search_hadits(client=mock_client, query="niat", top_k=5)
        assert resp.query == "niat"
        assert resp.total == 1

    def test_hasil_kosong(self, mock_model, mock_client):
        mock_client.search.return_value = make_opensearch_response([])
        resp = search_hadits(client=mock_client, query="tidak ada", top_k=5)
        assert resp.total == 0

    def test_mode_knn_default(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", top_k=5)
        body = mock_client.search.call_args.kwargs["body"]
        assert "knn" in body["query"]

    def test_mode_bm25(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", top_k=5, mode="bm25")
        body = mock_client.search.call_args.kwargs["body"]
        assert "multi_match" in body["query"]

    def test_mode_hybrid(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", top_k=5, mode="hybrid")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]


class TestAdvancedSearchHaditsService:

    def test_tanpa_filter_query_langsung(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="shalat", top_k=5, nama_perawi=None)
        body = mock_client.search.call_args.kwargs["body"]
        assert "knn" in body["query"]

    def test_dengan_filter_perawi_pakai_bool(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="shalat", top_k=5, nama_perawi="Bukhari")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]
        assert body["query"]["bool"]["filter"][0]["term"]["nama_perawi"] == "Bukhari"

    def test_return_search_response(self, mock_model, mock_client):
        resp = advanced_search_hadits(client=mock_client, query="niat", top_k=5)
        assert resp.query == "niat"

    def test_nama_perawi_kosong_tidak_difilter(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="zakat", top_k=5, nama_perawi=None)
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" not in body["query"]

    def test_mode_bm25_dengan_filter(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="zakat", top_k=5, nama_perawi="Muslim", mode="bm25")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]
        assert body["query"]["bool"]["filter"][0]["term"]["nama_perawi"] == "Muslim"

    def test_advanced_search_top_k_dibatasi_dengan_filter(self, mock_model, mock_client):
        mock_client.count.return_value = {"count": 2}

        advanced_search_hadits(
            client=mock_client,
            query="zakat",
            top_k=50,
            nama_perawi="Muslim",
            mode="knn",
        )

        body = mock_client.search.call_args.kwargs["body"]
        assert body["size"] == 2
        assert body["query"]["bool"]["filter"][0]["term"]["nama_perawi"] == "Muslim"
        assert body["query"]["bool"]["must"]["knn"]["embedding"]["k"] == 2


class TestSearchRouter:

    def test_search_berhasil(self, mock_client, test_client):
        with patch("app.routers.hadits_router.search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse, HaditsResult
            mock_svc.return_value = SearchResponse(
                query="niat", total=1,
                results=[HaditsResult(
                    nama_perawi="Bukhari", nomor_hadits=1,
                    referensi_lengkap="Hadits Bukhari Nomor 1",
                    arab="...", terjemahan="...", score=0.9,
                )],
            )
            resp = test_client.post("/hadits/search", json={"query": "niat ibadah"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

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

    def test_search_mode_invalid_return_422(self, test_client):
        resp = test_client.post("/hadits/search", json={"query": "niat ibadah", "mode": "tidak_ada"})
        assert resp.status_code == 422


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
