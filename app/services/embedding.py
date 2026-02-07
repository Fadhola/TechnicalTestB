import random

class EmbeddingService:
    def fake_embed(self, text: str) -> list[float]:
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)]