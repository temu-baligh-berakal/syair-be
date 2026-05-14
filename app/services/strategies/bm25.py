from app.services.strategies import QueryStrategy, register_strategy

SOURCE_FIELDS = ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"]


@register_strategy("bm25")
class Bm25Strategy(QueryStrategy):
    """Pencarian keyword menggunakan BM25 full-text search."""

    def build_query(self, query_text: str, embedding: list[float], page: int, page_size: int) -> dict:
        from_index = (page - 1) * page_size
        return {
            "from": from_index,
            "size": page_size,
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["terjemahan^2", "arab"],
                    "type": "best_fields",
                }
            },
            "_source": SOURCE_FIELDS,
        }
