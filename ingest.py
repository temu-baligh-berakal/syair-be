# ingest.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from opensearchpy import helpers
from tqdm import tqdm
import time
import re

from app.config import INDEX_NAME
from app.services.opensearch_client import get_opensearch_client

def create_index(client, index_name):
    """Membuat index dengan engine lucene (OpenSearch 3.0+ compatible)"""
    index_body = {
        "settings": {
            "index.knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "nama_perawi": {"type": "keyword"}, 
                "nomor_hadits": {"type": "integer"}, 
                "referensi_lengkap": {"type": "keyword"},
                "arab": {"type": "text"},
                "terjemahan": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil", # Gunakan l2 atau cosinesimil
                        "engine": "lucene",           # FIX: Ganti nmslib ke lucene
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16
                        }
                    }
                }
            }
        }
    }

    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' sudah ada. Menghapus index lama...")
        client.indices.delete(index=index_name)
    
    client.indices.create(index=index_name, body=index_body)
    print(f"Index '{index_name}' berhasil dibuat dengan engine Lucene!")

def run_etl():
    print("1. Memuat Model AI (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("2. Membaca dan Membersihkan Dataset...")
    df = pd.read_csv('dataset/Sunnah.csv')
    
    # Drop baris kosong
    df = df.dropna(subset=['Terjemahan', 'Arab', 'Perawi'])
    
    # --- PROSES EKSTRAKSI REGEX ---
    # Asumsi format: 'Hadits {Perawi} Nomor {i}'
    # Regex ini akan menangkap kata/kalimat di tengah (nama_perawi) dan angka di akhir (nomor)
    extracted = df['Perawi'].str.extract(r'(?i)Hadits\s+(.*?)\s+Nomor\s+(\d+)')
    
    # Memasukkan hasil ekstraksi ke kolom baru
    df['nama_perawi'] = extracted[0].fillna("Tidak Diketahui")
    df['nomor_hadits'] = extracted[1].fillna(0).astype(int)
    # ------------------------------

    df['Arab'] = df['Arab'].astype(str)
    df['Terjemahan'] = df['Terjemahan'].astype(str)
    
    total_docs = len(df)
    print(f"Total dokumen siap diproses: {total_docs}")

    client = get_opensearch_client()
    create_index(client, INDEX_NAME)

    print("3. Memulai proses Embedding dan Bulk Indexing...")
    batch_size = 256
    
    start_time = time.time()
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Ingesting Data"):
        batch_df = df.iloc[i:i+batch_size]
        
        teks_list = batch_df['Terjemahan'].tolist()
        embeddings = model.encode(teks_list, show_progress_bar=False)
        
        actions = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "nama_perawi": row['nama_perawi'],
                    "nomor_hadits": row['nomor_hadits'],
                    "referensi_lengkap": row['Perawi'], # Tetap simpan aslinya misal "Hadits Bukhari Nomor 1"
                    "arab": row['Arab'],
                    "terjemahan": row['Terjemahan'],
                    "embedding": embeddings[j].tolist()
                }
            }
            actions.append(action)
            
        helpers.bulk(client, actions)

    print(f"\nSelesai! Waktu proses: {(time.time() - start_time):.2f} detik.")

if __name__ == "__main__":
    run_etl()