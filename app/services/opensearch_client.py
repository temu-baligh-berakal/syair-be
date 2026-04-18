# app/services/opensearch_client.py
from opensearchpy import OpenSearch
from ..config import OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASS

# Singleton client — dibuat sekali, dipakai berulang
_client: OpenSearch | None = None


def get_opensearch_client() -> OpenSearch:
    """Mengembalikan OpenSearch client (singleton)."""
    global _client
    if _client is None:
        _client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
    return _client