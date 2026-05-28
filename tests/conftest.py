# tests/conftest.py
import os

# Set environment variable palsu di sini
# Pytest akan mengeksekusi ini sebelum meng-import module apapun dari aplikasi
os.environ["GROQ_API_KEY_1"] = "dummy_mock_key_untuk_testing_saja"
os.environ["FEEDBACK_SQLITE_PATH"] = "/tmp/syair_test_search_feedback.sqlite3"
os.environ["RERANKER_PROVIDER"] = "local"
os.environ.pop("JINA_API_KEY", None)
