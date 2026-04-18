from app.services.strategies import QueryStrategy, register_strategy

SOURCE_FIELDS = ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"]


@register_strategy("knn")
class KnnStrategy(QueryStrategy):
    """Pencarian semantik murni menggunakan KNN vector similarity."""

    def build_query(self, query_text: str, embedding: list[float], top_k: int) -> dict:
        return {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k,
                    }
                }
            },
            "_source": SOURCE_FIELDS,
        }
