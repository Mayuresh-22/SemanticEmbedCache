# this script benchmarks the SemanticEmbedCache performance and 
# LangChain's CacheBackedEmbeddings performance for comparison

import csv
import os
import time
from langchain_cohere import CohereEmbeddings
import tqdm
from src.const.const import SIMILARITY_THRESHOLD
from src.SemanticEmbedCache import SemanticEmbedCache
from src.embedder.KeyEmbedder import KeyEmbedder
from src.storage.InMemStorage import InMemStorage

from dotenv import load_dotenv
load_dotenv(override=True)

os.environ["COHERE_API_KEY"] = str(os.getenv("COHERE_API_KEY"))
os.environ["BENCHMARKING"] = "1"

def log(message: str) -> None:
    with open(f"benchmark_log_{SIMILARITY_THRESHOLD}.txt", "a") as f:
        f.write(f"[BENCHMARK] {message}\n")

def main():
    log(f"**Embedding + SEC benchmark (threshold: {SIMILARITY_THRESHOLD})**")
    og_embedder = CohereEmbeddings(model="embed-english-v3.0")  # type: ignore
    
    key_embedder = KeyEmbedder()
    storage = InMemStorage()
    sec = SemanticEmbedCache(
        key_embedder=key_embedder,
        og_embedder=og_embedder,
        storage=storage
    )
    
    with open("benchmark_queries.csv", "r") as f:
        query_dataset = csv.reader(f)
        next(query_dataset, None)
        
        sec_times = []
        sec_hits = 0
        for row in tqdm.tqdm(query_dataset, desc="Benchmarking SEC", unit="query"):
            query = row[1]
            st = time.time()
            embedd, is_hit = sec._benchmark_get(query)
            et = time.time()
            sec_times.append(et - st)
            sec_hits += 1 if is_hit else 0
            time.sleep(2)
        
    avg_sec_time = sum(sec_times) / len(sec_times)
    hit_rate = sec_hits / len(sec_times) * 100
    
    log(f"SEC avg embedd time: {avg_sec_time} seconds")
    log(f"SEC hit rate: {hit_rate} %")
    
    time.sleep(60)
    
    log("**Embedding + LangChain's CacheBackedEmbeddings benchmark**")
    
    from langchain_classic.embeddings import CacheBackedEmbeddings
    from langchain_core.stores import InMemoryBaseStore
    
    store = InMemoryBaseStore()
    langchain_embedd_cache = CacheBackedEmbeddings(
        og_embedder,
        document_embedding_store=store,
        query_embedding_store=store
    )
    
    with open("benchmark_queries.csv", "r") as f:
        query_dataset = csv.reader(f)
        next(query_dataset, None)
        
        lec_times = []
        lec_hits = 0
        for row in tqdm.tqdm(query_dataset, desc="Benchmarking LangChain's CacheBackedEmbeddings", unit="query"):
            query = row[1]
            st = time.time()
            embedd, is_hit = langchain_embedd_cache.embed_query(query)
            et = time.time()
            lec_times.append(et - st)
            lec_hits += 1 if is_hit else 0
            time.sleep(2)
            # Note: LangChain's CacheBackedEmbeddings does not provide hit/miss info directly
            # So I've made custom modifications to return that info for benchmarking purposes.
    
    avg_lec_time = sum(lec_times) / len(lec_times)
    lec_hit_rate = lec_hits / len(lec_times) * 100
    
    log(f"LangChain's CacheBackedEmbeddings avg embedd time: {avg_lec_time} seconds")
    log(f"LangChain's CacheBackedEmbeddings hit rate: {lec_hit_rate} %")
    log("\n**Benchmarking completed**")

if __name__ == "__main__":
    main()    