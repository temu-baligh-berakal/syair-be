from fastapi import FastAPI
from app.routers.hadits_router import router as hadits_router
from app.routers.llm_summarizer_router import router as llm_summarizer_router

app = FastAPI(title="Syair API", version="0.1.0")

app.include_router(hadits_router)
app.include_router(llm_summarizer_router, prefix="/llm", tags=["LLM Summarizer"])


@app.get("/")
def root():
    return {"message": "Syair API is running"}
