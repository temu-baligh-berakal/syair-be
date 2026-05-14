import os
import sys
from pprint import pprint

sys.path.append("/home/rafli/Projects/syair/syair-be")

from app.services.opensearch_client import get_opensearch_client
from app.services.hadits_service import search_hadits

def test_search():
    client = get_opensearch_client()
    query = "kucing itu najis"
    for mode in ["knn", "bm25", "hybrid"]:
        res = search_hadits(client, query, top_k=5, mode=mode)
        print(f"--- Mode: {mode} ---")
        for r in res.results:
            print(f"Score: {r.score:.4f} | {r.referensi_lengkap}")

if __name__ == "__main__":
    test_search()
