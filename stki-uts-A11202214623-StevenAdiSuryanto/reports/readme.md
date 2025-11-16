```
Struktur projek:
    ├───app             -> berisi chat.py untuk antarmuka (CLI based)
    ├───data            -> berisi data yang digunakan
    │   └───processed   -> berisi data setelah lewat processs preprocessing
    ├───notebooks       -> berisi jupyter notebook
    │   ├───dev-only    -> bagian ini hanya sebagai building-block saja
    │   └───images      -> kumpulan gambar yang digunakan dalam notebook utama
    ├───reports         -> berisi informasi untuk menjalankan projek ini
    └───src             -> berisi source code berupa python script
```

NOTE:
  - `requirements.txt` menjamin bahwa seluruh *.py dan notebook utama dapat dijalankan, namun tidak menjamin notebook di dev-only dapat dijalankan.
  - Projek ini dirancang untuk bisa dijalankan lewat 3 jalan (meskipun fokusnya berbeda-beda):
    1. UTS_STKI_a11202214623.ipynb => fokus pada penjelasan 
    2. chat.py                     => fokus pada penggunaan engine sebagai sebuah aplikasi (tidak dapat kustomisasi)
    3. search_engine.py            => fokus pada penggunaan engine lewat eksekusi command (sehingga opsi lebih lengkap daripada lewat chat.py)