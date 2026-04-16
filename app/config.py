# config.py
import os
from dotenv import load_dotenv

# WAJIB: Memuat file .env yang ada di root direktori
load_dotenv() 

# Pastikan mengambil nilai string, buang spasi ekstra dengan .strip()
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost").strip()
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin").strip()
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "admin").strip()

INDEX_NAME = "sunnah_index"