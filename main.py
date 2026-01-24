import os
import time
from langchain_cohere import CohereEmbeddings
from langchain_core.embeddings import FakeEmbeddings
from src.SemanticEmbedCache import SemanticEmbedCache
from src.embedder.KeyEmbedder import KeyEmbedder
from src.storage.InMemStorage import InMemStorage

from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["COHERE_API_KEY"] = str(os.getenv("COHERE_API_KEY"))

def main():
    print("Hello from SemanticEmbedCache!")
    key_embedder = KeyEmbedder()
    og_embedder = CohereEmbeddings(model="embed-english-v3.0")  # type: ignore
    storage = InMemStorage()
    
    sec = SemanticEmbedCache(
        key_embedder=key_embedder,
        og_embedder=og_embedder,
        storage=storage
    )
    
    st = time.time()
    sec._benchmark_get("How do I reset my password?")
    et = time.time()
    print(f"First call took {et - st} seconds.")
    st2 = time.time()
    sec._benchmark_get("I forgot my password")
    et2 = time.time()
    print(f"Second call took {et2 - st2} seconds.")


if __name__ == "__main__":
    main()
