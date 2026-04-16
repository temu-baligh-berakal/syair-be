from pydantic import BaseModel

class HaditsResponse(BaseModel):
    nama_perawi: str
    nomor_hadits: int
    referensi_lengkap: str
    arab: str
    terjemahan: str
    score: float # Nilai kemiripan dari OpenSearch