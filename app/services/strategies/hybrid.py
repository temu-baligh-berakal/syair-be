from app.services.strategies import QueryStrategy, register_strategy

SOURCE_FIELDS = ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"]


@register_strategy("hybrid")
class HybridStrategy(QueryStrategy):
    """Hybrid search: gabungan KNN (70%) dan BM25 (30%)."""

    def build_query(self, query_text: str, embedding: list[float], page: int, page_size: int) -> dict:
        from_index = (page - 1) * page_size
        return {
            "from": from_index,
            "size": page_size,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": from_index + page_size,
                                    "boost": 0.7,
                                }
                            }
                        },
                        {
                            "query_string": {
                                "query": query_text,
                                "fields": ["terjemahan^2", "arab"],
                                "boost": 0.3,
                                "default_operator": "AND",
                                "analyze_wildcard": True,
                                "allow_leading_wildcard": False,
                                "fuzziness": "AUTO",
                            }
                        },
                    ]
                }
            },
            "_source": SOURCE_FIELDS,
        }
