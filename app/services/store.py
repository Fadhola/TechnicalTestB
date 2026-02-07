import random
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

class DocumentStore:
    def __init__(self, qdrant_url="http://localhost:6333"):
        try:
            self.client = QdrantClient(qdrant_url)
            self.client.recreate_collection(
                collection_name="demo_collection",
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            self.use_qdrant = True
        except Exception:
            print("⚠️ Qdrant not available. Using in-memory store.")
            self.use_qdrant = False
            self.docs = []

    def add_document(self, text, vector):
        if self.use_qdrant:
            self.client.upsert(
                collection_name="demo_collection",
                points=[PointStruct(id=random.randint(1, 1e6), vector=vector, payload={"text": text})]
            )
        else:
            self.docs.append(text)

    def search(self, vector, query, limit=2):
        if self.use_qdrant:
            hits = self.client.search(collection_name="demo_collection", query_vector=vector, limit=limit)
            return [hit.payload["text"] for hit in hits]
        else:
            return [doc for doc in self.docs if query.lower() in doc.lower()]