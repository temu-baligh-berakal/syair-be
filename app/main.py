from fastapi import FastAPI
from app.routers.hadits_router import router as hadits_router

app = FastAPI(title="Syair API", version="0.1.0")

app.include_router(hadits_router)


@app.get("/")
def root():
    return {"message": "Syair API is running"}
