"""
Registry-based Strategy Pattern untuk query OpenSearch.

Tambah mode baru:
  1. Buat file baru di folder strategies/
  2. Buat class turunan QueryStrategy
  3. Dekorasi dengan @register_strategy("nama_mode")
  4. Selesai — tidak perlu ubah file lain.
"""
from abc import ABC, abstractmethod


class QueryStrategy(ABC):
    """Abstract base class untuk semua strategy pencarian."""

    @abstractmethod
    def build_query(
        self,
        query_text: str,
        embedding: list[float],
        top_k: int,
    ) -> dict:
        """Bangun OpenSearch query body."""
        ...


_registry: dict[str, QueryStrategy] = {}


def register_strategy(name: str):
    """
    Decorator untuk mendaftarkan strategy ke registry.

    Contoh:
        @register_strategy("knn")
        class KnnStrategy(QueryStrategy):
            ...
    """
    def decorator(cls):
        _registry[name] = cls()
        return cls
    return decorator


def get_strategy(name: str) -> QueryStrategy:
    """Ambil strategy berdasarkan nama. Raise ValueError jika tidak ditemukan."""
    if name not in _registry:
        available = ", ".join(_registry.keys())
        raise ValueError(f"Mode '{name}' tidak dikenal. Tersedia: {available}")
    return _registry[name]


def get_available_modes() -> list[str]:
    """Return daftar mode yang sudah terdaftar."""
    return list(_registry.keys())
