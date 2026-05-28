import os
import psycopg2
from psycopg2.extras import execute_values
from opensearchpy import OpenSearch, helpers
from dotenv import load_dotenv
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor

# Tambahkan root directory ke sys.path agar bisa import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.opensearch_client import get_opensearch_client
from app.config import INDEX_NAME

load_dotenv()

def init_db_structure():
    """Inisialisasi tabel di Neon DB."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL tidak ditemukan di .env")
        sys.exit(1)
        
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    # Reset data: Hapus tabel lama jika ada
    print("Resetting database: Dropping table hadits_similarity...")
    cur.execute("DROP TABLE IF EXISTS hadits_similarity;")
    
    cur.execute("""
        CREATE TABLE hadits_similarity (
            source_id TEXT,
            target_id TEXT,
            score FLOAT,
            PRIMARY KEY (source_id, target_id)
        );
        CREATE INDEX idx_source_id ON hadits_similarity(source_id);
    """)
    conn.commit()
    conn.close()

def worker_process(docs_batch):
    """Fungsi yang dijalankan oleh setiap worker untuk memproses satu batch dokumen."""
    if not docs_batch:
        return 0
        
    # Setiap worker butuh client dan koneksi DB sendiri (Process safe)
    client = get_opensearch_client()
    database_url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    data_to_insert = []
    
    try:
        for doc in docs_batch:
            source_id = doc['_id']
            embedding = doc['_source']['embedding']
            
            try:
                # Cari 11 teratas (1 dirinya sendiri + 10 yang mirip)
                res = client.search(index=INDEX_NAME, body={
                    "size": 11,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": 11
                            }
                        }
                    }
                })
                
                hits = res['hits']['hits']
                for hit in hits:
                    target_id = hit['_id']
                    if source_id == target_id: continue
                    
                    data_to_insert.append((source_id, target_id, hit['_score']))
                
                # Batch insert setiap 500 records untuk efisiensi
                if len(data_to_insert) >= 500:
                    execute_values(
                        cur, 
                        "INSERT INTO hadits_similarity (source_id, target_id, score) VALUES %s ON CONFLICT DO NOTHING", 
                        data_to_insert
                    )
                    data_to_insert = []
                    conn.commit()
                
            except Exception as e:
                print(f"\nError processing {source_id}: {e}")
                
        # Insert sisa data
        if data_to_insert:
            execute_values(
                cur, 
                "INSERT INTO hadits_similarity (source_id, target_id, score) VALUES %s ON CONFLICT DO NOTHING", 
                data_to_insert
            )
            conn.commit()
    finally:
        cur.close()
        conn.close()
        
    return len(docs_batch)

def chunk_list(lst, n):
    """Membagi list menjadi n bagian."""
    if not lst:
        return []
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def run():
    # 1. Siapkan struktur tabel
    init_db_structure()
    
    client = get_opensearch_client()
    
    # 2. Ambil semua metadata (ID dan Embedding) dari OpenSearch
    print("Mengambil daftar hadits dari OpenSearch...")
    query = {"_source": ["embedding"], "query": {"match_all": {}}}
    docs = list(helpers.scan(client, index=INDEX_NAME, query=query))
    total_docs = len(docs)
    
    if total_docs == 0:
        print("Tidak ada dokumen ditemukan di index.")
        return

    num_workers = 12
    print(f"Memproses {total_docs} hadits dengan {num_workers} workers secara paralel...")
    
    # 3. Bagi data menjadi 12 bagian untuk 12 workers
    chunks = chunk_list(docs, num_workers)
    
    # 4. Jalankan secara paralel menggunakan ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Menggunakan tqdm untuk memantau penyelesaian setiap worker
        list(tqdm(executor.map(worker_process, chunks), total=len(chunks), desc="Parallel Workers Progress"))
    
    print("\nSelesai! Seluruh data kemiripan telah disimpan ke Neon DB.")

if __name__ == "__main__":
    run()
