from typing import Optional, Tuple
import numpy as np
from fastembed.common.types import NumpyArray
from langchain_core.embeddings import Embeddings

from src.const.const import HIGHEST_SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD
from src.embedder.BaseEmbedder import BaseEmbedder
from src.embedder.KeyEmbedder import KeyEmbedder
from src.storage.BaseStorage import BaseStorage


class SemanticEmbedCache:
    def __init__(
        self,
        key_embedder: KeyEmbedder,
        og_embedder: BaseEmbedder | Embeddings,
        storage: BaseStorage,
    ):
        """
        Initializes the empty SemanticEmbedCache with the given key embedder,
        original embedder, and storage.
        
        :param self: 
        :param key_embedder: KeyEmbedder instance to generate key embeddings
        :type key_embedder: KeyEmbedder
        :param og_embedder: Original embedder to generate feature-rich and dense embeddings
        :type og_embedder: BaseEmbedder | Embeddings
        :param storage: Storage instance to store and retrieve embeddings
        :type storage: BaseStorage
        """
        self.key_embedder = key_embedder
        self.og_embedder = og_embedder
        self.storage = storage
        self._deserialized_keys_cache: dict[str, NumpyArray] = {}
        self._serialize_key = lambda key_embedding: np.array_str(key_embedding[0])
        self._deserialize_key = lambda serialized_key: np.fromstring(
            serialized_key.strip("[]"), dtype=np.float64, sep=" "
        )
        self._cosine_similarity = lambda a, b: (
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        )
        
    def get(self, text: str) -> list[float]:
        """
        Embeds the given text using the semantic embed cache.
        
        This method first generates a computationally inexpensive key
        embedding for the input text. It then checks the storage for an exact
        match. If no exact match is found, it performs a similarity search
        among stored keys. If a similar key is found, whose similarity score
        exceeds a predefined threshold, the corresponding cached embedding is
        returned. If neither an exact nor a similar match is found, the method
        uses original embedder to compute the embedding, stores it in the cache, and returns it.
        
        :param self:
        :param text: String to be embedded
        :type text: str
        :return: Embedded vector as a list of floats
        :rtype: list[float]
        """
        
        key_embedding = self.key_embedder.embed_key(text)
        serialized_key = self._serialize_key(key_embedding)

        # direct lookup for exact key embedding match
        cached_embedding = self.storage.get(serialized_key)
        if cached_embedding is not None:
            return cached_embedding

        # similarity search over all the available keys
        similar_key = self._similarity_search(key_embedding)
        if similar_key is not None:
            return self.storage.get(similar_key)

        og_embedding = self._embedd_text(text)
        self.storage.set(serialized_key, og_embedding)
        return og_embedding

    def _benchmark_get(self, text: str) -> Tuple[list[float], bool]:
        """
        It's the same as "get" method but also returns whether it was a cache hit or miss.
        Also, it prints "cache hit" when there is a cache hit.
        
        :param self: 
        :param text: String to be embedded
        :type text: str
        :return: Tuple containing the embedded vector as a list of floats and a boolean indicating cache hit or miss
        :rtype: Tuple[list[float], bool]
        """
        key_embedding = self.key_embedder.embed_key(text)
        serialized_key = self._serialize_key(key_embedding)

        # direct lookup for exact key embedding match
        cached_embedding = self.storage.get(serialized_key)
        if cached_embedding is not None:
            print("cache hit")
            return cached_embedding, True

        # similarity search over all the available keys
        similar_key = self._similarity_search(key_embedding)
        if similar_key is not None:
            print("cache hit")
            return self.storage.get(similar_key), True

        og_embedding = self._embedd_text(text)
        self.storage.set(serialized_key, og_embedding)
        return og_embedding, False

    def _embedd_text(self, text: str) -> list[float]:
        """
        Wrapper method to embed text using the original embedder
        
        :param self:
        :param text: String to be embedded
        :type text: str
        :return: Embedded vector as a list of floats
        :rtype: list[float]
        """
        if isinstance(self.og_embedder, BaseEmbedder):
            og_embedding = self.og_embedder.embed(text)
        elif isinstance(self.og_embedder, Embeddings):
            og_embedding = self.og_embedder.embed_query(text)
        else:
            raise ValueError(
                "og_embedder must be an instance of BaseEmbedder or Embeddings"
            )
        return og_embedding

    def _similarity_search(self, key_embedding: list[NumpyArray]) -> Optional[str]:
        """
        Helper method to perform similarity search over stored keys
        
        :param self:
        :param key_embedding: Key embedding to search for similar stored keys
        :type key_embedding: list[NumpyArray]
        :return: Stored key that is most similar to the given key embedding, or None if no similar key is found
        :rtype: str | None
        """
        max_score = -1.0
        stored_search_key = None
        for stored_key in self.storage.get_all_keys():
            if stored_key in self._deserialized_keys_cache:
                deserialized_key = self._deserialized_keys_cache[stored_key]
            else:
                deserialized_key = self._deserialize_key(stored_key)
                self._deserialized_keys_cache[stored_key] = deserialized_key
            sim_score = self._cosine_similarity(key_embedding[0], deserialized_key)
            if sim_score > SIMILARITY_THRESHOLD and sim_score > max_score:
                max_score = sim_score
                stored_search_key = stored_key
                if sim_score > HIGHEST_SIMILARITY_THRESHOLD:
                    break
        return stored_search_key
