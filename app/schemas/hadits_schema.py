from pydantic import BaseModel, Field, field_validator
from typing import Optional

from app.services.strategies import get_available_modes


class SearchQuery(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Teks pencarian dalam bahasa Indonesia",
        examples=["shalat berjamaah lebih utama dari shalat sendirian"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah hasil yang dikembalikan (1-50)",
    )
    nama_perawi: Optional[str] = Field(
        default=None,
        description="Filter berdasarkan nama perawi, misal: 'Bukhari', 'Muslim'",
        examples=["Bukhari"],
    )
    mode: str = Field(
        default="knn",
        description="Mode pencarian — nilai yang tersedia diambil dari registry strategy",
        examples=["knn", "bm25", "hybrid"],
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        available = get_available_modes()
        if v not in available:
            raise ValueError(f"Mode '{v}' tidak tersedia. Pilihan: {available}")
        return v


class HaditsResult(BaseModel):
    nama_perawi: str = Field(description="Nama perawi hadits, misal: 'Bukhari'")
    nomor_hadits: int = Field(description="Nomor hadits dalam kitab")
    referensi_lengkap: str = Field(
        description="Referensi lengkap, misal: 'Hadits Bukhari Nomor 1'"
    )
    arab: str = Field(description="Teks Arab hadits")
    terjemahan: str = Field(description="Terjemahan hadits dalam bahasa Indonesia")
    score: float = Field(description="Skor kemiripan semantik dari OpenSearch (0.0-1.0)")


class SearchResponse(BaseModel):
    query: str = Field(description="Query yang dikirim user")
    total: int = Field(description="Jumlah hasil yang ditemukan")
    results: list[HaditsResult] = Field(description="Daftar hadits yang relevan")