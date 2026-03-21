import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, collection_name: str = "rag_collection"):
        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        self.collection_name = collection_name

        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    @property
    def collection(self):
        """Dynamically get or create collection to handle deletion/recreation."""
        return self.client.get_or_create_collection(name=self.collection_name)

    def clear_collection(self):
        """Delete all documents from the collection and recreate it."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        # Accessing .collection property will recreate it automatically

    def add_documents(self, documents):
        """
        documents: list of dicts
        Each dict must be:
        {
            "content": str,
            "metadata": dict
        }
        """

        if not documents:
            return

        # Generate unique IDs
        start_id = self.collection.count()
        ids = [str(start_id + i) for i in range(len(documents))]

        texts = []
        metadatas = []

        for doc in documents:
            text = doc.get("content", "").strip()
            if not text:
                continue

            texts.append(text)

            # Clean metadata (Chroma only allows primitive types)
            clean_meta = {}
            for k, v in doc.get("metadata", {}).items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)

            metadatas.append(clean_meta)

        if not texts:
            return

        embeddings = self.embedding_model.encode(texts).tolist()

        self.collection.add(
            ids=ids[:len(texts)],
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query: str, k: int = 5):
        """
        Returns:
        {
            "documents": [[str, ...]],
            "metadatas": [[dict, ...]],
            "distances": [[float, ...]]
        }
        """

        if self.collection.count() == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )

        return {
            "documents": results.get("documents", [[]]),
            "metadatas": results.get("metadatas", [[]]),
            "distances": results.get("distances", [[]])
        }
