FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models agar termuat ke dalam Docker Image
# Ini mencegah server men-download ratusan MB saat pertama kali container jalan
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
