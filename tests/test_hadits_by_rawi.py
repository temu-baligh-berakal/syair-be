import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app as fastapi_app
from app.routers.hadits_router import get_client
from app.config import INDEX_NAME
from app.schemas.hadits_schema import ByRawiParams, ByRawiResponse, HaditsResult
from app.services.hadits_service import get_hadits_by_rawi


# ============================================================
# Helpers & Fixtures
# ============================================================


def make_opensearch_response(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


RAW_FAKE_HITS = [
    {
        "_score": 0.95,
        "_source": {
            "nama_perawi": "Bukhari",
            "nomor_hadits": 1,
            "referensi_lengkap": "Hadits Bukhari Nomor 1",
            "arab": "\u0625\u0650\u0646\u064e\u0651\u0645\u064e\u0627 \u0627\u0644\u0652\u0623\u064e\u0639\u0652\u0645\u064e\u0627\u0644\u064f \u0628\u0650\u0627\u0644\u0646\u0650\u0651\u064a\u064e\u0651\u0627\u062a\u0650",
            "terjemahan": "Sesungguhnya setiap amalan tergantung pada niatnya.",
        },
    },
    {
        "_score": 0.90,
        "_source": {
            "nama_perawi": "Bukhari",
            "nomor_hadits": 2,
            "referensi_lengkap": "Hadits Bukhari Nomor 2",
            "arab": "...",
            "terjemahan": "Terjemahan hadits kedua.",
        },
    },
]


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.count.return_value = {"count": 5}
    client.search.return_value = make_opensearch_response(RAW_FAKE_HITS)
    return client


@pytest.fixture
def mock_client_kosong():
    client = MagicMock()
    client.count.return_value = {"count": 0}
    client.search.return_value = make_opensearch_response([])
    return client


@pytest.fixture
def test_client(mock_client):
    fastapi_app.dependency_overrides[get_client] = lambda: mock_client
    yield TestClient(fastapi_app)
    fastapi_app.dependency_overrides.clear()


# ============================================================
# Schema: ByRawiParams
# ============================================================


class TestByRawiParams:
    def test_default_values(self):
        p = ByRawiParams()
        assert p.page == 1
        assert p.page_size == 10

    def test_custom_values(self):
        p = ByRawiParams(page=2, page_size=5)
        assert p.page == 2
        assert p.page_size == 5

    def test_page_minimum(self):
        p = ByRawiParams(page=1)
        assert p.page == 1

    def test_page_size_minimum(self):
        p = ByRawiParams(page_size=1)
        assert p.page_size == 1

    def test_page_size_maximum(self):
        p = ByRawiParams(page_size=50)
        assert p.page_size == 50

    def test_page_kurang_dari_1_gagal(self):
        with pytest.raises(ValidationError):
            ByRawiParams(page=0)

    def test_page_negatif_gagal(self):
        with pytest.raises(ValidationError):
            ByRawiParams(page=-1)

    def test_page_size_kurang_dari_1_gagal(self):
        with pytest.raises(ValidationError):
            ByRawiParams(page_size=0)

    def test_page_size_lebih_dari_50_gagal(self):
        with pytest.raises(ValidationError):
            ByRawiParams(page_size=51)

    def test_page_tipe_string_gagal(self):
        with pytest.raises(ValidationError):
            ByRawiParams(page="bukan_angka")


# ============================================================
# Schema: ByRawiResponse
# ============================================================


class TestByRawiResponse:
    def test_valid_single_result(self):
        results = [
            HaditsResult(
                nama_perawi="Bukhari",
                nomor_hadits=1,
                referensi_lengkap="Hadits Bukhari Nomor 1",
                arab="...",
                terjemahan="...",
                preview="...",
                score=0.95,
            ),
        ]
        resp = ByRawiResponse(
            rawi="Bukhari", total=1, page=1, page_size=10, results=results
        )
        assert resp.rawi == "Bukhari"
        assert resp.total == 1
        assert resp.page == 1
        assert resp.page_size == 10
        assert len(resp.results) == 1

    def test_valid_empty_results(self):
        resp = ByRawiResponse(rawi="Muslim", total=0, page=1, page_size=10, results=[])
        assert resp.total == 0
        assert resp.results == []

    def test_valid_multiple_results(self):
        results = [
            HaditsResult(
                nama_perawi="Bukhari",
                nomor_hadits=i,
                referensi_lengkap=f"Hadits Bukhari Nomor {i}",
                arab="...",
                terjemahan=f"Terjemahan {i}",
                preview=f"Preview {i}",
                score=0.9,
            )
            for i in range(1, 6)
        ]
        resp = ByRawiResponse(
            rawi="Bukhari", total=50, page=1, page_size=10, results=results
        )
        assert resp.total == 50
        assert len(resp.results) == 5
        assert resp.results[0].nomor_hadits == 1
        assert resp.results[4].nomor_hadits == 5

    def test_rawi_field_wajib(self):
        with pytest.raises(ValidationError):
            ByRawiResponse(total=1, page=1, page_size=10, results=[])

    def test_total_field_wajib(self):
        with pytest.raises(ValidationError):
            ByRawiResponse(rawi="Bukhari", page=1, page_size=10, results=[])

    def test_page_field_wajib(self):
        with pytest.raises(ValidationError):
            ByRawiResponse(rawi="Bukhari", total=1, page_size=10, results=[])

    def test_page_size_field_wajib(self):
        with pytest.raises(ValidationError):
            ByRawiResponse(rawi="Bukhari", total=1, page=1, results=[])

    def test_results_field_wajib(self):
        with pytest.raises(ValidationError):
            ByRawiResponse(rawi="Bukhari", total=1, page=1, page_size=10)


# ============================================================
# Service: get_hadits_by_rawi
# ============================================================


class TestByRawiService:
    def test_count_dipanggil_dengan_filter_term_rawi(self, mock_client):
        get_hadits_by_rawi(client=mock_client, rawi="Bukhari", page=1, page_size=10)
        mock_client.count.assert_called_once()
        body = mock_client.count.call_args.kwargs["body"]
        assert body["query"]["term"]["nama_perawi"] == "Bukhari"

    def test_search_body_mengandung_term_sort_pagination(self, mock_client):
        get_hadits_by_rawi(client=mock_client, rawi="Bukhari", page=1, page_size=10)
        mock_client.search.assert_called_once()
        body = mock_client.search.call_args.kwargs["body"]
        assert body["query"] == {"term": {"nama_perawi": "Bukhari"}}
        assert body["sort"] == [{"nomor_hadits": "asc"}]
        assert body["from"] == 0
        assert body["size"] == 10

    def test_pagination_halaman_pertama(self, mock_client):
        get_hadits_by_rawi(client=mock_client, rawi="Bukhari", page=1, page_size=10)
        body = mock_client.search.call_args.kwargs["body"]
        assert body["from"] == 0
        assert body["size"] == 10

    def test_pagination_halaman_kedua(self, mock_client):
        get_hadits_by_rawi(client=mock_client, rawi="Bukhari", page=2, page_size=5)
        body = mock_client.search.call_args.kwargs["body"]
        assert body["from"] == 5
        assert body["size"] == 5

    def test_pagination_halaman_besar(self, mock_client):
        mock_client.count.return_value = {"count": 100}
        get_hadits_by_rawi(client=mock_client, rawi="Bukhari", page=10, page_size=10)
        body = mock_client.search.call_args.kwargs["body"]
        assert body["from"] == 90
        assert body["size"] == 10

    def test_return_by_rawi_response(self, mock_client):
        resp = get_hadits_by_rawi(
            client=mock_client, rawi="Bukhari", page=1, page_size=10
        )
        assert isinstance(resp, ByRawiResponse)
        assert resp.rawi == "Bukhari"
        assert resp.total == 5
        assert resp.page == 1
        assert resp.page_size == 10
        assert len(resp.results) == 2
        assert resp.results[0].nama_perawi == "Bukhari"
        assert resp.results[0].nomor_hadits == 1

    def test_rawi_tidak_ditemukan_return_kosong(self, mock_client_kosong):
        resp = get_hadits_by_rawi(
            client=mock_client_kosong, rawi="RawiTidakAda", page=1, page_size=10
        )
        assert resp.total == 0
        assert resp.results == []
        assert resp.rawi == "RawiTidakAda"

    def test_zero_documents_tidak_panggil_search(self, mock_client_kosong):
        get_hadits_by_rawi(
            client=mock_client_kosong, rawi="TidakAda", page=1, page_size=10
        )
        mock_client_kosong.search.assert_not_called()

    def test_rawi_dengan_spasi(self, mock_client):
        get_hadits_by_rawi(
            client=mock_client, rawi="Abu Hurairah", page=1, page_size=10
        )
        body = mock_client.count.call_args.kwargs["body"]
        assert body["query"]["term"]["nama_perawi"] == "Abu Hurairah"

    def test_sort_ascending_by_nomor_hadits(self, mock_client):
        get_hadits_by_rawi(client=mock_client, rawi="Bukhari", page=1, page_size=10)
        body = mock_client.search.call_args.kwargs["body"]
        assert body["sort"] == [{"nomor_hadits": "asc"}]

    def test_rawi_empty_string(self, mock_client):
        mock_client.count.return_value = {"count": 0}
        resp = get_hadits_by_rawi(client=mock_client, rawi="", page=1, page_size=10)
        assert resp.total == 0
        assert resp.results == []

    def test_handle_score_null(self, mock_client):
        mock_client.search.return_value = make_opensearch_response(
            [
                {
                    "_score": None,
                    "_source": {
                        "nama_perawi": "Bukhari",
                        "nomor_hadits": 1,
                        "referensi_lengkap": "Hadits Bukhari Nomor 1",
                        "arab": "...",
                        "terjemahan": "Terjemahan.",
                    },
                },
            ]
        )
        resp = get_hadits_by_rawi(
            client=mock_client, rawi="Bukhari", page=1, page_size=10
        )
        assert len(resp.results) == 1
        assert resp.results[0].score == 0.0


# ============================================================
# Router: GET /hadits/by-rawi/{rawi}
# ============================================================


class TestByRawiRouter:
    def test_get_by_rawi_berhasil_200(self, test_client):
        with patch("app.routers.hadits_router.get_hadits_by_rawi") as mock_svc:
            mock_svc.return_value = ByRawiResponse(
                rawi="Bukhari",
                total=1,
                page=1,
                page_size=10,
                results=[
                    HaditsResult(
                        nama_perawi="Bukhari",
                        nomor_hadits=1,
                        referensi_lengkap="Hadits Bukhari Nomor 1",
                        arab="...",
                        terjemahan="...",
                        preview="...",
                        score=0.9,
                    )
                ],
            )
            resp = test_client.get("/hadits/by-rawi/Bukhari")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rawi"] == "Bukhari"
        assert data["total"] == 1
        assert len(data["results"]) == 1

    def test_get_by_rawi_empty_results_200(self, test_client):
        with patch("app.routers.hadits_router.get_hadits_by_rawi") as mock_svc:
            mock_svc.return_value = ByRawiResponse(
                rawi="TidakAda", total=0, page=1, page_size=10, results=[]
            )
            resp = test_client.get("/hadits/by-rawi/TidakAda")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_query_params_default(self, test_client):
        with patch("app.routers.hadits_router.get_hadits_by_rawi") as mock_svc:
            mock_svc.return_value = ByRawiResponse(
                rawi="Bukhari", total=0, page=1, page_size=10, results=[]
            )
            test_client.get("/hadits/by-rawi/Bukhari")
            _, kwargs = mock_svc.call_args
            assert kwargs["rawi"] == "Bukhari"
            assert kwargs["page"] == 1
            assert kwargs["page_size"] == 10

    def test_query_params_custom(self, test_client):
        with patch("app.routers.hadits_router.get_hadits_by_rawi") as mock_svc:
            mock_svc.return_value = ByRawiResponse(
                rawi="Muslim", total=0, page=2, page_size=5, results=[]
            )
            test_client.get("/hadits/by-rawi/Muslim?page=2&page_size=5")
            _, kwargs = mock_svc.call_args
            assert kwargs["rawi"] == "Muslim"
            assert kwargs["page"] == 2
            assert kwargs["page_size"] == 5

    def test_page_out_of_range_422(self, test_client):
        resp = test_client.get("/hadits/by-rawi/Bukhari?page=0")
        assert resp.status_code == 422

    def test_page_size_out_of_range_422(self, test_client):
        resp = test_client.get("/hadits/by-rawi/Bukhari?page_size=0")
        assert resp.status_code == 422

    def test_page_size_too_large_422(self, test_client):
        resp = test_client.get("/hadits/by-rawi/Bukhari?page_size=51")
        assert resp.status_code == 422

    def test_error_opensearch_500(self, test_client):
        with patch(
            "app.routers.hadits_router.get_hadits_by_rawi",
            side_effect=Exception("Connection refused"),
        ):
            resp = test_client.get("/hadits/by-rawi/Bukhari")
        assert resp.status_code == 500

    def test_rawi_dengan_spasi_url_encoded(self, test_client):
        with patch("app.routers.hadits_router.get_hadits_by_rawi") as mock_svc:
            mock_svc.return_value = ByRawiResponse(
                rawi="Abu Hurairah", total=0, page=1, page_size=10, results=[]
            )
            resp = test_client.get("/hadits/by-rawi/Abu%20Hurairah")
        assert resp.status_code == 200
        _, kwargs = mock_svc.call_args
        assert kwargs["rawi"] == "Abu Hurairah"
