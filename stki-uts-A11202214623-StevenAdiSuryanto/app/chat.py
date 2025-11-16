import sys
import os

# Add ../src to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from search_engine import load_preprocess_documents, build_vsm_engine, run_vsm


def generate_response(query, results):
    """
    Membuat jawaban interaktif berbasis template dari hasil VSM.
    results = [(doc_name, score), ...]
    """
    if not results:
        print(f"Tidak ada dokumen yang relevan untuk query: '{query}'.")
        return

    response = []
    response.append(f"🔍 Hasil pencarian untuk: `{query}`")
    response.append(f"Menampilkan {len(results)} dokumen paling relevan:\n")

    # Detail per dokumen
    for i, (doc, score) in enumerate(results, start=1):
        response.append(f"{i}. 📄 **{doc}** — skor relevansi: {score:.4f}")

    # Ambil 5 kata kunci dari query (manual rule-based)
    keywords = query.lower().split()
    top_keywords = ", ".join(keywords[:5])

    response.append("\n✨ **Kata kunci utama:** " + top_keywords)

    # Tambahan penutup
    response.append("\n🧠 Gunakan query lain untuk eksplorasi lebih dalam!")

    print("\n".join(response))


def simple_chat():
    print("=== Simple Search Chat ===")
    print(
        "Algoritma yang dipakai hanya VSM. Jika ingin mencoba algoritma Boolean, silakan eksekusi secara langsung script /src/search_engine.py"
    )
    print("Type 'exit' to quit.\n")

    docs = load_preprocess_documents()
    engine = build_vsm_engine(docs, k=3)

    while True:
        query = input("Masukkan query: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        result = run_vsm(query, engine)

        generate_response(query, result)


if __name__ == "__main__":
    simple_chat()
