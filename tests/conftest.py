# tests/conftest.py
import os

# Set environment variable palsu di sini
# Pytest akan mengeksekusi ini sebelum meng-import module apapun dari aplikasi
os.environ["GROQ_API_KEY_1"] = "dummy_mock_key_untuk_testing_saja"