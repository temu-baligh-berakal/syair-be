import pandas as pd
import re
from collections import Counter

def count_narrators():
    print("Membaca dataset/Sunnah.csv...")
    try:
        # Membaca CSV
        df = pd.read_csv('dataset/Sunnah.csv')
        
        narrators = []
        print("Mengekstrak nama perawi dari kolom Terjemahan...")
        
        # Regex untuk mencari teks di dalam kurung siku [...]
        # Pola: \[([^\]]+)\]
        pattern = re.compile(r'\[([^\]]+)\]')
        
        for text in df['Terjemahan']:
            if pd.isna(text):
                continue
            
            # Cari semua kemunculan [...] dalam satu baris terjemahan
            matches = pattern.findall(str(text))
            narrators.extend(matches)
        
        # Hitung frekuensi
        counts = Counter(narrators)
        
        # Urutkan berdasarkan jumlah (terbesar ke terkecil)
        sorted_counts = counts.most_common()
        
        # Simpan ke file .txt
        output_file = 'daftar_perawi.txt'
        print(f"Menyimpan hasil ke {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for name, count in sorted_counts:
                f.write(f"{name}: {count}\n")
        
        print(f"Selesai! Berhasil mengekstrak {len(counts)} perawi unik.")
        
    except FileNotFoundError:
        print("Error: File dataset/Sunnah.csv tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    count_narrators()
