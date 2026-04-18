import pytest
from pydantic import ValidationError

# Import strategies agar terdaftar di registry
import app.services.strategies.knn     # noqa: F401
import app.services.strategies.bm25    # noqa: F401
import app.services.strategies.hybrid  # noqa: F401

from app.schemas.hadits_schema import SearchQuery, HaditsResult, SearchResponse


class TestSearchQuery:

    def test_valid_query_minimal(self):
        q = SearchQuery(query="shalat berjamaah")
        assert q.query == "shalat berjamaah"
        assert q.top_k == 10
        assert q.nama_perawi is None
        assert q.mode == "knn"

    def test_valid_query_lengkap(self):
        q = SearchQuery(query="keutamaan puasa", top_k=5, nama_perawi="Bukhari", mode="bm25")
        assert q.top_k == 5
        assert q.nama_perawi == "Bukhari"
        assert q.mode == "bm25"

    def test_query_terlalu_pendek(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchQuery(query="ab")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)

    def test_query_terlalu_panjang(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="a" * 501)

    def test_query_wajib_ada(self):
        with pytest.raises(ValidationError):
            SearchQuery()

    def test_top_k_minimum(self):
        q = SearchQuery(query="zakat fitrah", top_k=1)
        assert q.top_k == 1

    def test_top_k_maksimum(self):
        q = SearchQuery(query="zakat fitrah", top_k=50)
        assert q.top_k == 50

    def test_top_k_nol_gagal(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="zakat fitrah", top_k=0)

    def test_top_k_lebih_dari_50_gagal(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="zakat fitrah", top_k=51)

    def test_nama_perawi_opsional_none(self):
        q = SearchQuery(query="sedekah jariyah")
        assert q.nama_perawi is None

    def test_nama_perawi_diisi(self):
        q = SearchQuery(query="sedekah jariyah", nama_perawi="Muslim")
        assert q.nama_perawi == "Muslim"

    def test_mode_default_knn(self):
        q = SearchQuery(query="sedekah jariyah")
        assert q.mode == "knn"

    def test_mode_bm25_valid(self):
        q = SearchQuery(query="sedekah jariyah", mode="bm25")
        assert q.mode == "bm25"

    def test_mode_hybrid_valid(self):
        q = SearchQuery(query="sedekah jariyah", mode="hybrid")
        assert q.mode == "hybrid"

    def test_mode_tidak_terdaftar_gagal(self):
        with pytest.raises(ValidationError) as exc_info:
            SearchQuery(query="sedekah jariyah", mode="tidak_ada")
        assert "tidak tersedia" in str(exc_info.value)


HADITS_VALID = {
    "nama_perawi": "Bukhari",
    "nomor_hadits": 1,
    "referensi_lengkap": "Hadits Bukhari Nomor 1",
    "arab": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ",
    "terjemahan": "Sesungguhnya setiap amalan tergantung pada niatnya.",
    "score": 0.95,
}

class TestHaditsResult:

    def test_valid_hadits(self):
        h = HaditsResult(**HADITS_VALID)
        assert h.nama_perawi == "Bukhari"
        assert h.nomor_hadits == 1
        assert h.score == 0.95

    def test_semua_field_wajib(self):
        for field in HADITS_VALID:
            data = {k: v for k, v in HADITS_VALID.items() if k != field}
            with pytest.raises(ValidationError):
                HaditsResult(**data)

    def test_nomor_hadits_harus_integer(self):
        with pytest.raises(ValidationError):
            HaditsResult(**{**HADITS_VALID, "nomor_hadits": "satu"})

    def test_score_harus_float(self):
        h = HaditsResult(**{**HADITS_VALID, "score": 1})
        assert isinstance(h.score, float)

class TestSearchResponse:

    def test_valid_response(self):
        resp = SearchResponse(
            query="niat",
            total=1,
            results=[HaditsResult(**HADITS_VALID)],
        )
        assert resp.total == 1
        assert len(resp.results) == 1
        assert resp.results[0].nama_perawi == "Bukhari"

    def test_response_kosong(self):
        resp = SearchResponse(query="tidak ada", total=0, results=[])
        assert resp.total == 0
        assert resp.results == []

    def test_response_banyak_hasil(self):
        results = [HaditsResult(**{**HADITS_VALID, "nomor_hadits": i, "score": 0.9 - i * 0.01})
                   for i in range(1, 6)]
        resp = SearchResponse(query="shalat", total=5, results=results)
        assert resp.total == 5
        assert resp.results[0].nomor_hadits == 1
        assert resp.results[4].nomor_hadits == 5

    def test_query_dan_total_wajib(self):
        with pytest.raises(ValidationError):
            SearchResponse(results=[])
