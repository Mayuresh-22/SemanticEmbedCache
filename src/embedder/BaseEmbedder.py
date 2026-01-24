class BaseEmbedder:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Class 'BaseEmbedder' is not implemented")
