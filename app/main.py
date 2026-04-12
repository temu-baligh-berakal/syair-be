from fastapi import FastAPI

app = FastAPI(title="Syair API", version="0.1.0")


@app.get("/")
def root():
    return {"message": "Syair API is running"}
