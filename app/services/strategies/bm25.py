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
                "query_string": {
                    "query": query_text,
                    "fields": ["terjemahan^2", "arab"],
                    "default_operator": "AND",
                    "analyze_wildcard": True,
                    "allow_leading_wildcard": False,
                    "fuzziness": "AUTO",
                }
            },
            "_source": SOURCE_FIELDS,
        }
