from app.services.strategies import QueryStrategy, register_strategy

SOURCE_FIELDS = ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"]


@register_strategy("hybrid")
class HybridStrategy(QueryStrategy):
    """Hybrid search: gabungan KNN (70%) dan BM25 (30%)."""

    def build_query(self, query_text: str, embedding: list[float], top_k: int) -> dict:
        return {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": top_k,
                                    "boost": 0.7,
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["terjemahan^2", "arab"],
                                "boost": 0.3,
                            }
                        },
                    ]
                }
            },
            "_source": SOURCE_FIELDS,
        }
