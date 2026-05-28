# Syair Hadits Search Engine

Sistem temu-balik informasi (information retrieval) yang dikhususkan untuk pencarian hadits berbahasa Indonesia, dibangun dengan FastAPI, OpenSearch, dan Next.js.

**Anggota:**
- Muhammad Raihan Maulana (2306216636)
- Muhammad Rafli Esa Pradana (2306207480)
- Raden Ahmad Yasin Mahendra (2306215154)

## Deployment

Link Deployment: syair.site

## Gambaran Umum Proyek

Syair Search Engine adalah sistem temu-balik informasi yang dirancang untuk pencarian hadits dalam bahasa Indonesia. Sistem ini menggunakan dataset **Sunnah** (`dataset/Sunnah_v2.csv`) untuk menyajikan hasil pencarian yang akurat dan relevan dari berbagai perawi seperti Bukhari, Muslim, Tirmidzi, Abu Daud, Nasai, Ibnu Majah, Ahmad, Malik, dan Darimi.

## Fitur Utama

### Pencarian Multi-Mode
- **BM25 (Lexical Search):** Pencocokan berbasis kata kunci untuk kueri dengan frasa yang spesifik.
- **KNN (Semantic Search):** Pencarian semantik berbasis vektor menggunakan embedding `paraphrase-multilingual-MiniLM-L12-v2` (dimensi 384, HNSW + engine Lucene).
- **Hybrid Search:** Penggabungan skor antara BM25 dan KNN untuk menyeimbangkan presisi leksikal dan recall semantik.

### Koreksi Kueri dan Saran
- Autocompletion real-time melalui `search_as_you_type` OpenSearch.
- Toleransi typo dengan saran "did you mean" ketika kepercayaan hasil rendah.

### Ringkasan Otomatis dengan AI
- Ringkasan singkat dari top-N hasil hadits menggunakan LLM melalui Groq API.
- Cache via Redis untuk mengoptimalkan latensi dan kuota penggunaan.

### Optimasi Domain Hadits
- Custom analyzer Bahasa Indonesia dengan filter sinonim Islami (`shalat/sholat/salat`, `hadits/hadis`, `wudhu/wudu`, `dzikir/zikir`, `shadaqah/sedekah`, `puasa/shaum`, `zakat/jakat`).
- Indonesian stemmer dan stopword filter.
- Pembersihan sanad saat ingest agar embedding fokus pada matan hadits.

### Reranking
- Default menggunakan **Jina Reranker API** (`jina-reranker-v3`) untuk reranking top-K kandidat.
- Fallback ke local cross-encoder (`cross-encoder/ms-marco-MiniLM-L2-v2`) yang di-preload saat server menyala apabila API key tidak tersedia atau provider mengalami rate limit.

### UI Responsif dan Modern
- Dibangun di atas Next.js 16, React 19, dan Tailwind CSS 4.
- Komponen Radix UI untuk dialog/select, animasi Framer Motion, dan toast Sonner.
- Pagination dan filter facet berdasarkan perawi..

## Arsitektur Sistem

Sistem terdiri dari beberapa komponen utama:

**Frontend (Next.js + Tailwind CSS)**
- Antarmuka web untuk input kueri, tampilan hasil, dan ringkasan AI.
- Autocompletion sisi klien dan dialog pengaturan pencarian.

**Backend (FastAPI + OpenSearch)**
- Endpoint REST API di bawah `/hadits` dan `/llm`.
- Logika indexing dokumen, retrieval, pemrosesan kueri, dan reranking.

**Semantic Engine**
- Sentence Transformers untuk embedding dokumen dan kueri.
- KNN vector search via OpenSearch Lucene HNSW.

**Integrasi LLM**
- Groq API untuk inferensi cepat pada ringkasan top-N hadits.
- Cache Redis untuk menghindari generasi yang berulang.

**Storage Layer**
- OpenSearch (`sunnah_index`) sebagai index pencarian utama.
- PostgreSQL via `psycopg2-binary` untuk data aplikasi.
- Redis untuk cache.
- SQLite untuk pencatatan feedback ringan.

## Petunjuk Setup

### Prasyarat
- Python 3.11 atau lebih tinggi
- Node.js 20 atau lebih tinggi (dengan `pnpm`)
- Docker (untuk OpenSearch / Redis)

### 1. Clone Repository
```bash
git clone <repo-url> syair
cd syair
```

Proyek dipisahkan ke dalam dua folder:
- [syair-be/](syair-be/) — Backend FastAPI
- [syair-fe/](syair-fe/) — Frontend Next.js

### 2. Setup Backend

#### Buat environment
```bash
cd syair-be

# Menggunakan venv
python -m venv venv
source venv/bin/activate     # macOS/Linux
# venv\Scripts\activate      # Windows

# Install PyTorch versi CPU terlebih dahulu (sesuai Dockerfile)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependency lainnya
pip install -r requirements.txt
```

#### Konfigurasi environment variables
Buat file `.env` di dalam [syair-be/](syair-be/):
```
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASS=admin

GROQ_API_KEY=your_groq_api_key_here
JINA_API_KEY=your_jina_api_key_here
RERANKER_PROVIDER=jina
```

#### Jalankan OpenSearch
```bash
docker pull opensearchproject/opensearch:latest

docker run -d -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "plugins.security.disabled=true" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=Syair123!" \
  opensearchproject/opensearch:latest
```

#### Indexing dataset
```bash
python ingest.py --dataset dataset/Sunnah_v2.csv
```

Proses ini akan:
1. Memuat model multilingual MiniLM.
2. Membersihkan sanad dari tiap hadits agar embedding lebih bersih.
3. Bulk-indexing dokumen ke index `sunnah_index` dengan vektor HNSW.

#### Jalankan API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API akan tersedia di http://127.0.0.1:8000/.

### 3. Setup Frontend
```bash
cd syair-fe

pnpm install
pnpm dev
```

Frontend akan tersedia di http://127.0.0.1:3000/.

Sesuaikan URL backend di [syair-fe/.env](syair-fe/.env) jika berbeda dari default.

## Cara Menggunakan Search Engine

- **Pencarian Dasar:** Ketikkan kueri di kolom pencarian lalu submit.
- **Pemilihan Mode:** Pilih BM25, KNN, atau Hybrid melalui dialog pengaturan pencarian.
- **Autocomplete:** Saran muncul saat Anda mengetik.
- **Pagination dan Facet:** Persempit hasil berdasarkan perawi.
- **AI Summary:** Otomatis dihasilkan dari top hadits hasil pencarian.
- **Saran Typo:** Klik saran koreksi yang muncul.


## Struktur Proyek

```
syair/
├── advanced-hadits-search-roadmap.txt   # Roadmap peningkatan retrieval
├── syair-be/                            # Backend FastAPI
│   ├── app/
│   │   ├── config.py                    # Konfigurasi env + index
│   │   ├── main.py                      # FastAPI app + lifespan model loading
│   │   ├── routers/                     # hadits_router, llm_summarizer_router
│   │   ├── schemas/                     # Pydantic schemas
│   │   ├── services/                    # hadits_service, llm_summarizer, cache, feedback
│   │   │   └── strategies/              # Query builder BM25 / KNN / Hybrid
│   │   └── models/
│   ├── dataset/                         # Sunnah.csv, Sunnah_v2.csv
│   ├── lib/                             # Daftar perawi + helper
│   ├── scripts/                         # compute_similarity.py, rawi_scrap.py
│   ├── tests/                           # Pytest suite
│   ├── data/                            # search_feedback.sqlite3
│   ├── ingest.py                        # ETL: pembersihan sanad, embedding, bulk index
│   ├── requirements.txt
│   └── Dockerfile
├── syair-fe/                            # Frontend Next.js
│   ├── app/                             # App router pages, layout, search, hadits
│   ├── components/                      # LlmSummary, SearchResultItem, SearchSettingsDialog, ui/
│   ├── public/                          # Aset statis
│   ├── package.json
│   └── Dockerfile
└── README.md                            # File ini
```

## Troubleshooting

### Masalah Koneksi OpenSearch
```bash
# Cek apakah container berjalan
docker ps

# Restart jika perlu
docker restart <container_id>
```

Jika log API menampilkan error autentikasi, pastikan `OPENSEARCH_USER` dan `OPENSEARCH_PASS` di `.env` cocok dengan container yang berjalan.

### Error Ringkasan LLM
Jika AI summary gagal:
- Verifikasi `GROQ_API_KEY` di [syair-be/.env](syair-be/.env).
- Pastikan Redis dapat dijangkau jika caching diaktifkan.
- Periksa response endpoint `/llm` di log backend.

### Reranker Jina Tidak Aktif
Jika reranker fallback ke local atau menampilkan log cooldown:
- Verifikasi `JINA_API_KEY` di [syair-be/.env](syair-be/.env).
- Cek `RERANKER_PROVIDER=jina` aktif (default sudah jina).
- Jika rate limit, sistem otomatis cooldown selama `EXTERNAL_RERANKER_RATE_LIMIT_COOLDOWN_SECONDS` (default 60 detik) dan memakai cross-encoder lokal sementara.

### Model Download Lambat
Saat startup pertama, embedding model dan cross-encoder akan diunduh. Dockerfile sudah pre-download keduanya saat build:
```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
  SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
  CrossEncoder('cross-encoder/ms-marco-MiniLM-L2-v2')"
```

Untuk pengembangan lokal, jalankan snippet di atas sekali agar cache terisi.

### Masalah Instalasi Package
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

## Sumber Tambahan
- [Dokumentasi FastAPI](https://fastapi.tiangolo.com/)
- [Dokumentasi OpenSearch](https://opensearch.org/docs/latest/)
- [Dokumentasi Sentence Transformers](https://www.sbert.net/)
- [Dokumentasi Next.js](https://nextjs.org/docs)
- [Dokumentasi Groq API](https://console.groq.com/docs)

## Kontak

Jika Anda memiliki pertanyaan atau menemui kendala, silakan hubungi:
- Muhammad Raihan Maulana: muhammad.raihan@ui.ac.id
- Muhammad Rafli Esa Pradana: muhammad.rafli@ui.ac.id
- Raden Ahmad Yasin Mahendra: raden.ahmad@ui.ac.id
