from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers.hadits_router import router as hadits_router
from app.routers.llm_summarizer_router import router as llm_summarizer_router
from app.services.hadits_service import get_model, get_cross_encoder

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload model ML ke RAM saat server baru menyala
    print("Memuat model ML ke memori...")
    get_model()
    get_cross_encoder()
    print("Model ML berhasil dimuat!")
    yield
    # Proses pembersihan jika ada (shutdown)

app = FastAPI(title="Syair API", version="0.1.0", lifespan=lifespan)

app.include_router(hadits_router)
app.include_router(llm_summarizer_router, prefix="/llm", tags=["LLM Summarizer"])


@app.get("/")
def root():
    return {"message": "Syair API is running"}
