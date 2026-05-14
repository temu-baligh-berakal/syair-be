# ingest.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from opensearchpy import helpers
from tqdm import tqdm
import time
import re

from app.config import INDEX_NAME
from app.services.opensearch_client import get_opensearch_client

def extract_clean_text(text):
    """
    Ekstrak isi hadits dengan membuang sanad berdasarkan pola 
    tanda petik ganda (") ATAU kata transisi 'bahwa/sanya'.
    """
    if pd.isna(text):
        return ""
    
    # 1. Coba pola tanda petik (Prioritas Utama)
    match = re.search(r'"([^"]*)"', text)
    if match:
        return match.group(1).strip()
    
    # 2. Fallback ke pola "bahwa" jika tanda petik tidak ada
    if "bahwa" in text.lower() or "bahwasanya" in text.lower():
        parts = re.split(r'bahwasanya|bahwa', text, flags=re.IGNORECASE)
        return parts[-1].strip()
    
    # 3. Kembalikan teks asli jika tidak ada pola (kasus NEITHER)
    return text.strip()

def create_index(client, index_name):
    """Membuat index dengan engine lucene (OpenSearch 3.0+ compatible)"""
    index_body = {
        "settings": {
            "index.knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "islamic_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "shalat, sholat, salat",
                            "hadits, hadis",
                            "wudhu, wudu",
                            "dzikir, zikir",
                            "shadaqah, sedekah",
                            "puasa, shaum",
                            "zakat, jakat"
                        ]
                    },
                    "indonesian_stop": {
                        "type": "stop",
                        "stopwords": "_indonesian_"
                    },
                    "indonesian_stemmer": {
                        "type": "stemmer",
                        "language": "indonesian"
                    }
                },
                "analyzer": {
                    "indonesian_with_synonyms": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "islamic_synonyms",
                            "indonesian_stop",
                            "indonesian_stemmer"
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "nama_perawi": {"type": "keyword"}, 
                "nomor_hadits": {"type": "integer"}, 
                "referensi_lengkap": {"type": "keyword"},
                "arab": {"type": "text"},
                
                # --- PERUBAHAN UTAMA: CUSTOM ANALYZER DENGAN SINONIM & AUTOCOMPLETE ---
                "terjemahan": {
                    "type": "text",
                    "analyzer": "indonesian_with_synonyms",
                    "search_analyzer": "indonesian_with_synonyms",
                    "fields": {
                        "suggest": {
                            "type": "search_as_you_type"
                        }
                    }
                },
                # ----------------------------------------------------
                
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil", 
                        "engine": "lucene",           
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
    print(f"Index '{index_name}' berhasil dibuat dengan engine Lucene dan Indonesian Analyzer!")

def run_etl():
    print("1. Memuat Model AI (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("2. Membaca dan Membersihkan Dataset...")
    df = pd.read_csv('dataset/Sunnah.csv')
    
    # Drop baris kosong
    df = df.dropna(subset=['Terjemahan', 'Arab', 'Perawi'])
    
    # --- PROSES EKSTRAKSI REGEX PERAWI & NOMOR ---
    extracted = df['Perawi'].str.extract(r'(?i)Hadits\s+(.*?)\s+Nomor\s+(\d+)')
    df['nama_perawi'] = extracted[0].fillna("Tidak Diketahui")
    df['nomor_hadits'] = extracted[1].fillna(0).astype(int)
    
    # --- PROSES PEMBERSIHAN SANAD UNTUK EMBEDDING ---
    df['Arab'] = df['Arab'].astype(str)
    df['Terjemahan'] = df['Terjemahan'].astype(str)
    print("Membersihkan sanad dari teks terjemahan...")
    df['teks_bersih'] = df['Terjemahan'].apply(extract_clean_text)
    
    total_docs = len(df)
    print(f"Total dokumen siap diproses: {total_docs}")

    client = get_opensearch_client()
    create_index(client, INDEX_NAME)

    print("3. Memulai proses Embedding dan Bulk Indexing...")
    batch_size = 256
    
    start_time = time.time()
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Ingesting Data"):
        batch_df = df.iloc[i:i+batch_size]
        
        # KUNCI KNN: Embedding dilakukan pada teks yang sudah dibersihkan dari sanad
        teks_list = batch_df['teks_bersih'].tolist()
        embeddings = model.encode(teks_list, show_progress_bar=False)
        
        actions = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "nama_perawi": row['nama_perawi'],
                    "nomor_hadits": row['nomor_hadits'],
                    "referensi_lengkap": row['Perawi'], 
                    "arab": row['Arab'],
                    "terjemahan": row['Terjemahan'], # Teks utuh disimpan untuk Highlighting & BM25
                    "embedding": embeddings[j].tolist()
                }
            }
            actions.append(action)
            
        helpers.bulk(client, actions)

    print(f"\nSelesai! Waktu proses: {(time.time() - start_time):.2f} detik.")

if __name__ == "__main__":
    run_etl()