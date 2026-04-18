import pytest
from pydantic import ValidationError

from app.schemas.hadits_schema import SearchQuery, HaditsResult, SearchResponse


class TestSearchQuery:

    def test_valid_query_minimal(self):
        """Query valid dengan field wajib saja."""
        q = SearchQuery(query="shalat berjamaah")
        assert q.query == "shalat berjamaah"
        assert q.top_k == 10
        assert q.nama_perawi is None

    def test_valid_query_lengkap(self):
        """Query valid dengan semua field diisi."""
        q = SearchQuery(query="keutamaan puasa", top_k=5, nama_perawi="Bukhari")
        assert q.top_k == 5
        assert q.nama_perawi == "Bukhari"

    def test_query_terlalu_pendek(self):
        """Query kurang dari 3 karakter harus gagal."""
        with pytest.raises(ValidationError) as exc_info:
            SearchQuery(query="ab")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)

    def test_query_terlalu_panjang(self):
        """Query lebih dari 500 karakter harus gagal."""
        with pytest.raises(ValidationError):
            SearchQuery(query="a" * 501)

    def test_query_wajib_ada(self):
        """Field query tidak boleh kosong/tidak ada."""
        with pytest.raises(ValidationError):
            SearchQuery()

    def test_top_k_minimum(self):
        """top_k minimum adalah 1."""
        q = SearchQuery(query="zakat fitrah", top_k=1)
        assert q.top_k == 1

    def test_top_k_maksimum(self):
        """top_k maksimum adalah 50."""
        q = SearchQuery(query="zakat fitrah", top_k=50)
        assert q.top_k == 50

    def test_top_k_nol_gagal(self):
        """top_k = 0 harus gagal (ge=1)."""
        with pytest.raises(ValidationError):
            SearchQuery(query="zakat fitrah", top_k=0)

    def test_top_k_lebih_dari_50_gagal(self):
        """top_k > 50 harus gagal (le=50)."""
        with pytest.raises(ValidationError):
            SearchQuery(query="zakat fitrah", top_k=51)

    def test_nama_perawi_opsional_none(self):
        """nama_perawi boleh None (tidak diisi)."""
        q = SearchQuery(query="sedekah jariyah")
        assert q.nama_perawi is None

    def test_nama_perawi_diisi(self):
        """nama_perawi bisa diisi string."""
        q = SearchQuery(query="sedekah jariyah", nama_perawi="Muslim")
        assert q.nama_perawi == "Muslim"


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
        """HaditsResult valid dengan semua field terisi."""
        h = HaditsResult(**HADITS_VALID)
        assert h.nama_perawi == "Bukhari"
        assert h.nomor_hadits == 1
        assert h.score == 0.95

    def test_semua_field_wajib(self):
        """Semua field di HaditsResult wajib ada."""
        for field in HADITS_VALID:
            data = {k: v for k, v in HADITS_VALID.items() if k != field}
            with pytest.raises(ValidationError):
                HaditsResult(**data)

    def test_nomor_hadits_harus_integer(self):
        """nomor_hadits harus integer, bukan string acak."""
        with pytest.raises(ValidationError):
            HaditsResult(**{**HADITS_VALID, "nomor_hadits": "satu"})

    def test_score_harus_float(self):
        """score harus bisa diparse sebagai float."""
        h = HaditsResult(**{**HADITS_VALID, "score": 1})  # int → float auto-cast
        assert isinstance(h.score, float)

class TestSearchResponse:

    def test_valid_response(self):
        """SearchResponse valid dengan satu hasil."""
        resp = SearchResponse(
            query="niat",
            total=1,
            results=[HaditsResult(**HADITS_VALID)],
        )
        assert resp.total == 1
        assert len(resp.results) == 1
        assert resp.results[0].nama_perawi == "Bukhari"

    def test_response_kosong(self):
        """SearchResponse boleh memiliki results kosong."""
        resp = SearchResponse(query="tidak ada", total=0, results=[])
        assert resp.total == 0
        assert resp.results == []

    def test_response_banyak_hasil(self):
        """SearchResponse bisa menampung banyak HaditsResult."""
        results = [HaditsResult(**{**HADITS_VALID, "nomor_hadits": i, "score": 0.9 - i * 0.01})
                   for i in range(1, 6)]
        resp = SearchResponse(query="shalat", total=5, results=results)
        assert resp.total == 5
        assert resp.results[0].nomor_hadits == 1
        assert resp.results[4].nomor_hadits == 5

    def test_query_dan_total_wajib(self):
        """Field query dan total wajib ada di SearchResponse."""
        with pytest.raises(ValidationError):
            SearchResponse(results=[])
