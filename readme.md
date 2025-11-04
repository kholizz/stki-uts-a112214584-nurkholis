STKI UTS – Mini Search Engine (Boolean + VSM)

Cara Menjalankan di Komputer Lokal:
1. Siapkan lingkungan kerja (virtual environment) dan pasang semua dependensi:
   pip install -r requirements.txt

2. Siapkan data:
   Letakkan 5 sampai 15 file teks (.txt) di dalam folder data/raw/.

3. Jalankan proses pembersihan teks (preprocessing) di notebooks:
   import sys, os
   sys.path.append(os.path.abspath(".."))  # biar folder src bisa dikenali
   from src.preprocess import preprocess_folder
   report = preprocess_folder('../data/raw','../data/processed')
   report[:2]

4. Menjalankan pencarian Boolean:
   python -m src.search --model boolean --query "sistem AND informasi" --data data/processed

5. Menjalankan VSM (menampilkan 5 hasil teratas):
   python -m src.search --model vsm --k 5 --query "sistem informasi kampus" --data data/processed
   Jika ingin menggunakan pembobotan sublinear TF:
   python -m src.search --model vsm --k 5 --query "sistem informasi kampus" --data data/processed --sublinear

6. Menjalankan antarmuka chat sederhana (opsional):
   python -m app.main


Struktur Proyek:
- src/preprocess.py  → Modul pembersihan teks (case folding, tokenisasi, stopword removal, stemming)
- src/boolean_ir.py  → Membuat incidence matrix / inverted index dan menjalankan kueri Boolean
- src/vsm_ir.py      → Menghitung TF, DF, IDF, TF-IDF, cosine similarity, dan menentukan ranking top-k
- src/eval.py        → Fungsi evaluasi: precision, recall, F1, P@k, AP/MAP, nDCG
- src/search.py      → File utama (CLI) untuk menjalankan model pencarian
- app/main.py        → Antarmuka chat sederhana berbasis VSM
- notebooks/UTS_STKI_demo.ipynb → Notebook untuk uji coba dan eksperimen
- reports/laporan.pdf    → Draft laporan versi 6–10 halaman
- reports/SOAL1.pdf    → Draft laporan SOAL 1

Catatan:
Jika semua dokumen korpus berbahasa Indonesia, aktifkan fitur stemmer Bahasa Indonesia
dengan membuka file src/preprocess.py dan ubah baris:
   USE_ID_STEMMER = True
