# tests/conftest.py
import os
import pytest

# Set environment variable palsu di sini
# Pytest akan mengeksekusi ini sebelum meng-import module apapun dari aplikasi
os.environ["GROQ_API_KEY_1"] = "dummy_mock_key_untuk_testing_saja"
os.environ["FEEDBACK_SQLITE_PATH"] = "/tmp/syair_test_search_feedback.sqlite3"
os.environ["RERANKER_PROVIDER"] = "local"
os.environ["REDIS_URL"] = ""
os.environ.pop("JINA_API_KEY", None)


@pytest.fixture(autouse=True)
def reset_search_cache_singleton():
    from app.services import search_cache_service

    search_cache_service._redis_client = None
    search_cache_service._redis_checked = False
    yield
    search_cache_service._redis_client = None
    search_cache_service._redis_checked = False
