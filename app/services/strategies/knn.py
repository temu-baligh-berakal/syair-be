from app.services.strategies import QueryStrategy, register_strategy

SOURCE_FIELDS = ["nama_perawi", "nomor_hadits", "referensi_lengkap", "arab", "terjemahan"]


@register_strategy("knn")
class KnnStrategy(QueryStrategy):
    """Pencarian semantik murni menggunakan KNN vector similarity."""

    def build_query(self, query_text: str, embedding: list[float], page: int, page_size: int) -> dict:
        from_index = (page - 1) * page_size
        return {
            "from": from_index,
            "size": page_size,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": from_index + page_size,
                    }
                }
            },
            "_source": SOURCE_FIELDS,
        }
