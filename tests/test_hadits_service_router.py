import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.routers.hadits_router import get_client
from app.services.hadits_service import _parse_hit, search_hadits, advanced_search_hadits, get_suggestions
from app.services.strategies import get_strategy, get_available_modes



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
    return {"hits": {"total": {"value": len(hits)}, "hits": hits}}


@pytest.fixture
def mock_model():
    with patch("app.services.hadits_service.get_model") as m:
        model = MagicMock()
        model.encode.return_value = FAKE_EMBEDDING
        m.return_value = model
        yield model


@pytest.fixture(autouse=True)
def mock_cross_encoder():
    with patch("app.services.hadits_service.get_cross_encoder") as m:
        ce_model = MagicMock()
        ce_model.predict.side_effect = lambda pairs: [1.0] * len(pairs)
        m.return_value = ce_model
        yield ce_model


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
            body = strategy.build_query(query_text="test", embedding=embedding, page=2, page_size=5)
            assert "query" in body
            assert "from" in body
            assert body["from"] == 5
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


class TestParseSuggestion:
    def test_parse_suggestion_ada(self):
        from app.services.hadits_service import _parse_suggestion
        suggest_block = {
            "spell_check": [
                {"text": "shlat", "options": [{"text": "shalat"}]},
                {"text": "brjamaah", "options": [{"text": "berjamaah"}]}
            ]
        }
        res = _parse_suggestion(suggest_block)
        assert res == "shalat berjamaah"

    def test_parse_suggestion_sebagian_ada(self):
        from app.services.hadits_service import _parse_suggestion
        suggest_block = {
            "spell_check": [
                {"text": "shalat", "options": []},
                {"text": "brjamaah", "options": [{"text": "berjamaah"}]}
            ]
        }
        res = _parse_suggestion(suggest_block)
        assert res == "shalat berjamaah"

    def test_parse_suggestion_tidak_ada_typo(self):
        from app.services.hadits_service import _parse_suggestion
        suggest_block = {
            "spell_check": [
                {"text": "shalat", "options": []},
            ]
        }
        res = _parse_suggestion(suggest_block)
        assert res is None


class TestSearchHaditsService:

    def test_memanggil_model_encode(self, mock_model, mock_cross_encoder, mock_client):
        search_hadits(client=mock_client, query="niat ibadah", page=1, page_size=5)
        mock_model.encode.assert_called_once_with("niat ibadah")

    def test_memanggil_opensearch_search(self, mock_model, mock_cross_encoder, mock_client):
        search_hadits(client=mock_client, query="niat ibadah", page=1, page_size=5)
        mock_client.search.assert_called_once()

    def test_pagination_parameter_from_and_size(self, mock_model, mock_client):
        # Gunakan use_reranker=False agar backend mem-passing 'from' langsung ke OpenSearch query (Tanpa fetch_size 30 statis)
        search_hadits(client=mock_client, query="niat ibadah", page=3, page_size=10, use_reranker=False)
        body = mock_client.search.call_args.kwargs["body"]
        assert body["from"] == 20
        assert body["size"] == 10

        if "knn" in body["query"]:
            assert body["query"]["knn"]["embedding"]["k"] == 30

    def test_threshold_min_score_applied(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", threshold=0.75)
        body = mock_client.search.call_args.kwargs["body"]
        assert "min_score" in body
        assert body["min_score"] == 0.75

    def test_return_search_response(self, mock_model, mock_client):
        mock_client.search.return_value = {
            "hits": {"total": {"value": 1}, "hits": [FAKE_HIT]},
            "suggest": {"spell_check": [{"text": "shlat", "options": [{"text": "shalat"}]}]}
        }
        resp = search_hadits(client=mock_client, query="shlat", page_size=5)
        assert resp.query == "shlat"
        assert resp.total == 1
        assert resp.suggestion == "shalat"

    def test_hasil_kosong(self, mock_model, mock_client):
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
        resp = search_hadits(client=mock_client, query="tidak ada", page_size=5)
        assert resp.total == 0

    def test_mode_knn_default(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", page_size=5)
        body = mock_client.search.call_args.kwargs["body"]
        assert "knn" in body["query"]

    def test_mode_bm25(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", page_size=5, mode="bm25")
        body = mock_client.search.call_args.kwargs["body"]
        assert "multi_match" in body["query"]

    def test_query_prosedural_mererank_hasil_bertata_cara(self, mock_model, mock_client):
        procedural_hit = {
            "_score": 1.2,
            "_source": {
                "nama_perawi": "Muslim",
                "nomor_hadits": 248,
                "referensi_lengkap": "Hadits Muslim Nomor 248",
                "arab": "...",
                "terjemahan": "Kemudian beliau berwudhu seperti wudhu untuk shalat, lalu mengguyur kepala dan seluruh badannya.",
            },
        }
        non_procedural_hit = {
            "_score": 4.8,
            "_source": {
                "nama_perawi": "Bukhari",
                "nomor_hadits": 127,
                "referensi_lengkap": "Hadits Bukhari Nomor 127",
                "arab": "...",
                "terjemahan": "Apabila perempuan melihat air maka dia wajib mandi.",
            },
        }
        mock_client.search.return_value = {
            "hits": {"total": {"value": 2}, "hits": [non_procedural_hit, procedural_hit]},
        }

        resp = search_hadits(client=mock_client, query="bagaimana cara mandi wajib", page_size=5)

        assert resp.results[0].nama_perawi == "Muslim"
        assert "berwudhu" in resp.results[0].terjemahan.lower()


class TestAutocompleteSuggestion:

    def test_get_suggestions_memakai_highlight_fragment(self, mock_client):
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "terjemahan": "Rumah tempat tinggal anak lelaki pamanku sangat luas.",
                        },
                        "highlight": {
                            "terjemahan": [
                                "Keutamaan menuntut **ilmu** dan mengamalkannya",
                            ]
                        },
                    }
                ],
            }
        }

        suggestions = get_suggestions(mock_client, "ilmu")

        assert suggestions == ["ilmu dan mengamalkannya"]

    def test_get_suggestions_fallback_ke_snippet_match(self, mock_client):
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "terjemahan": "Orang yang memiliki keutamaan sabar akan memperoleh pahala besar.",
                        },
                    }
                ],
            }
        }

        suggestions = get_suggestions(mock_client, "keutamaan")

        assert suggestions == ["keutamaan sabar akan memperoleh"]

    def test_get_suggestions_menyelesaikan_kata_prefix(self, mock_client):
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "terjemahan": "Keutamaan menuntut ilmu dan mengamalkannya sangat besar.",
                        },
                    }
                ],
            }
        }

        suggestions = get_suggestions(mock_client, "keuta")

        assert suggestions == ["keutamaan menuntut ilmu dan"]

    def test_get_suggestions_buang_kandidat_yang_tidak_mengandung_seluruh_frasa(self, mock_client):
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "terjemahan": "Keutamaan bubur atas semua makanan.",
                        },
                    },
                    {
                        "_source": {
                            "terjemahan": "Keutamaan Allah atas makhluknya.",
                        },
                    },
                ],
            }
        }

        suggestions = get_suggestions(mock_client, "keutamaan solat berjamaah")

        assert suggestions == []

    def test_mode_hybrid(self, mock_model, mock_client):
        search_hadits(client=mock_client, query="niat", page_size=5, mode="hybrid")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]


class TestAdvancedSearchHaditsService:

    def test_tanpa_filter_query_langsung(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="shalat", page_size=5, nama_perawi=None)
        body = mock_client.search.call_args.kwargs["body"]
        assert "knn" in body["query"]

    def test_dengan_filter_perawi_pakai_bool(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="shalat", page_size=5, nama_perawi="Bukhari")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]
        assert body["query"]["bool"]["filter"][0]["term"]["nama_perawi"] == "Bukhari"

    def test_return_search_response(self, mock_model, mock_client):
        mock_client.search.return_value = {"hits": {"total": {"value": 1}, "hits": [FAKE_HIT]}}
        resp = advanced_search_hadits(client=mock_client, query="niat", page_size=5)
        assert resp.query == "niat"

    def test_nama_perawi_kosong_tidak_difilter(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="zakat", page_size=5, nama_perawi=None)
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" not in body["query"]

    def test_mode_bm25_dengan_filter(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="zakat", page_size=5, nama_perawi="Muslim", mode="bm25")
        body = mock_client.search.call_args.kwargs["body"]
        assert "bool" in body["query"]
        assert body["query"]["bool"]["filter"][0]["term"]["nama_perawi"] == "Muslim"

    def test_advanced_threshold_min_score_applied(self, mock_model, mock_client):
        advanced_search_hadits(client=mock_client, query="niat", threshold=0.85)
        body = mock_client.search.call_args.kwargs["body"]
        assert "min_score" in body
        assert body["min_score"] == 0.85


class TestSearchRouter:

    def test_search_berhasil(self, mock_client, test_client):
        with patch("app.routers.hadits_router.search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse, HaditsResult
            mock_svc.return_value = SearchResponse(
                query="niat", total=1,
                results=[HaditsResult(
                    nama_perawi="Bukhari", nomor_hadits=1,
                    referensi_lengkap="Hadits Bukhari Nomor 1",
                    arab="...", terjemahan="...", 
                    preview="...",
                    score=0.9,
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

    def test_advanced_search_pagination_dikirim(self, mock_client, test_client):
        with patch("app.routers.hadits_router.advanced_search_hadits") as mock_svc:
            from app.schemas.hadits_schema import SearchResponse
            mock_svc.return_value = SearchResponse(query="zakat", total=0, results=[])
            test_client.post("/hadits/advanced-search", json={"query": "zakat fitrah", "page": 2, "page_size": 15})
            _, kwargs = mock_svc.call_args
            assert kwargs.get("page") == 2
            assert kwargs.get("page_size") == 15

    def test_advanced_search_error_return_500(self, mock_client, test_client):
        with patch("app.routers.hadits_router.advanced_search_hadits", side_effect=Exception("timeout")):
            resp = test_client.post("/hadits/advanced-search", json={"query": "shalat malam"})
        assert resp.status_code == 500
