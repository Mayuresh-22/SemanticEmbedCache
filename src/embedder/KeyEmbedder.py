from fastembed import TextEmbedding
from fastembed.common.types import NumpyArray


class KeyEmbedder:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-en"):
        self.key_embedder = TextEmbedding(model_name=model_name)

    def embed_key(self, key: str) -> list[NumpyArray]:
        return list(self.key_embedder.query_embed(key))
