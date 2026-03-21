
print("Importing imports...")
try:
    import fitz
    print("PyMuPDF loaded.")
except ImportError as e:
    print(f"FAILED PyMuPDF: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformers loaded.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")
    emb = model.encode(["test"])
    print(f"Embedding generated: {len(emb[0])} dim")
except Exception as e:
    print(f"FAILED SentenceTransformers: {e}")

try:
    import chromadb
    print("ChromaDB loaded.")
    client = chromadb.Client()
    print("Chroma Client created.")
except Exception as e:
    print(f"FAILED ChromaDB: {e}")

print("Done.")
