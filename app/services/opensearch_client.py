# app/services/opensearch_client.py
from opensearchpy import OpenSearch
from ..config import OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_USER, OPENSEARCH_PASS

def get_opensearch_client():
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_compress=True, # Mempercepat transfer data
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        use_ssl=True,       # Ubah ke False jika berjalan di localhost tanpa HTTPS
        verify_certs=False, # Ubah ke True jika di production
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client