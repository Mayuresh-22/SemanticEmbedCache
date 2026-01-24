# SemanticEmbedCache (SEC)
A semantic similarity-based cache for text embeddings that reduces redundant API calls and improves cache hit rates.

## Motivation
Current embedding cache implementations (LangChain, etc.) use **exact string matching** as cache keys. This means "How do I reset my password?" and "I forgot my password" are treated as completely different queries, causing cache misses even though they're semantically identical.

**Problems with exact string matching:**
- Sensitive to typos, punctuation, and minor wording changes
- Misses semantically similar queries
- Results in redundant API calls and increased costs
- Lower cache efficiency (~24% hit rate on varied queries)

**SemanticEmbedCache solution:**
Uses **semantic similarity search** to find cached embeddings for semantically similar texts, even when worded differently. This improves cache hit rates by 2-3x compared to exact matching.

## Features
- **Semantic similarity matching**: Finds cached embeddings using cosine similarity (threshold: 0.85)
- **Two-tier lookup**: Fast exact match, fallback to similarity search
- **Performance optimizations**: 
  - In-memory deserialized key cache
  - Early exit for high similarity (>0.98)
- **Pluggable architecture**: 
  - Compatible with any embedder implementing `BaseEmbedder` or LangChain's `Embeddings` interface
  - Swappable storage backends (via `BaseStorage` interface)
- **Easy integration**: Simple API with `.get(text)` method

## Architecture

### Cache Structure
SemanticEmbedCache stores pairs of (key embedding, original embedding):

| Key (Serialized Key Embedding) | Value (Original Embedding) |
|-------------------------------|---------------------------|
| String representation of key embedding | Embedding from original embedder |

**Components:**
- **Key Embedding**: Fast, lightweight embedding using FastEmbed's `jinaai/jina-embeddings-v2-base-en` (local model)
  - Used for similarity search only
  - Serialized as string for storage keys
- **Original Embedding**: Full embedding from your chosen embedder (e.g., Cohere, OpenAI)
  - The actual embedding returned to your application
  - Stored as value in cache

### Why Two Embeddings?
1. **Key embeddings** are cheap/fast (local FastEmbed) → used for finding similar queries
2. **Original embeddings** are expensive (API calls) → the actual feature-rich embeddings you and your application need
3. This 2-tier design minimizes API costs while enabling semantic search

## How It Works

```
Input text: "I forgot my password"
     ↓
1. Generate key embedding (FastEmbed - local, fast)
     ↓
2. Check exact match in cache (serialized key lookup)
     ↓ (miss)
3. Similarity search over all cached keys (cosine similarity)
     ↓ (found: "How do I reset my password?" with 0.87 similarity)
4. Return cached original embedding ✓ (cache hit!)

If no match found:
     ↓
5. Generate original embedding (API call - expensive)
     ↓
6. Store (key_embedding → original_embedding) in cache
     ↓
7. Return original embedding
```

### Lookup Flow
1. **Key embedding generation**: Compute FastEmbed embedding for input text
2. **Exact match lookup**: Check if serialized key embedding exists in storage → instant return if found
3. **Similarity search** (if no exact match):
   - Iterate through all stored keys
   - Compute cosine similarity between query key and each stored key
   - If similarity > `SIMILARITY_THRESHOLD` (0.85), return cached embedding
   - If similarity > `HIGHEST_SIMILARITY_THRESHOLD` (0.98), early exit (near-duplicate)
4. **Cache miss**: Generate original embedding via API, store in cache, return

## Benchmark Results

Tested on 100 diverse queries (exact duplicates, semantic variations, unique queries):

### Threshold: 0.85 (Recommended)
| Implementation | Hit Rate | Avg Time | Improvement |
|----------------|----------|----------|-------------|
| **SemanticEmbedCache** | **60.0%** | 0.300s | **2.5x better hit rate** |
| LangChain CacheBackedEmbeddings | 24.0% | 0.281s | Only exact matches |

### Threshold: 0.90 (Strict)
| Implementation | Hit Rate | Avg Time | Improvement |
|----------------|----------|----------|-------------|
| **SemanticEmbedCache** | **49.0%** | 0.327s | **2.04x better hit rate** |
| LangChain CacheBackedEmbeddings | 24.0% | 0.282s | Only exact matches |

**Key Findings:**
- **Hit Rate Impact**: Lowering threshold from 0.90 to 0.85 improves hit rate from 49% → 60% (+22% improvement)
- **Speed Trade-off**: SEC is ~7% slower due to similarity search, but this is offset by:
  - 2-3x fewer API calls (60% cache hits vs 24%)
  - Significant cost savings on embedding API usage
  - Overall faster application performance due to reduced network latency

**Threshold Selection Guide:**
- **0.90**: Strict matching, fewer false positives, 49% hit rate
- **0.85**: Balanced (recommended), good semantic matching, 60% hit rate  
- **0.80**: More lenient, higher hit rate but risk of unrelated matches

**Notes:**
- Average time includes embedding generation and cache lookup
- LangChain's CacheBackedEmbeddings doesn't return hit/miss status natively; benchmark uses custom modification
- Benchmarks performed using Cohere's `embed-english-v3.0` model as original embedder
- Dataset: 100 queries with ~30% exact duplicates, ~40% semantic variations, ~30% unique queries

## Configuration

```python
# src/const/const.py
SIMILARITY_THRESHOLD = 0.85           # Minimum similarity for cache hit
HIGHEST_SIMILARITY_THRESHOLD = 0.98   # Early exit threshold (near-duplicates)
```

## Usage Example

```python
from langchain_cohere import CohereEmbeddings
from src.SemanticEmbedCache import SemanticEmbedCache
from src.embedder.KeyEmbedder import KeyEmbedder
from src.storage.InMemStorage import InMemStorage

# Initialize components
key_embedder = KeyEmbedder()  # FastEmbed local model
og_embedder = CohereEmbeddings(model="embed-english-v3.0")
storage = InMemStorage()

# Create cache
sec = SemanticEmbedCache(
    key_embedder=key_embedder,
    og_embedder=og_embedder,
    storage=storage
)

# Use cache
embedding = sec.get("How do I reset my password?")  # API call (miss)
embedding = sec.get("I forgot my password")          # Cache hit! (0.87 similarity)
```

## Performance Optimization Opportunities

Current implementation uses **O(n) linear search** through all cached keys. For larger caches:

**Recommended optimization**: Add FAISS for approximate nearest neighbor search
- Reduces search complexity from O(n) to O(log n)
- Expected speedup: 10-100x for caches with 1000+ entries
- See implementation guide in codebase discussions

## API Reference

### `SemanticEmbedCache`

**`__init__(key_embedder, og_embedder, storage)`**
- `key_embedder`: `KeyEmbedder` instance for generating key embeddings
- `og_embedder`: `BaseEmbedder | Embeddings` - your original embedder (Cohere, OpenAI, etc.)
- `storage`: `BaseStorage` - storage backend (InMemStorage or custom)

**`get(text: str) -> list[float]`**
- Embeds text using semantic cache
- Returns: List of floats (the original embedding)

**`_benchmark_get(text: str) -> Tuple[list[float], bool]`**
- Same as `get()` but also returns cache hit/miss status
- For benchmarking purposes only

## Extending SemanticEmbedCache

### Custom Storage Backend
Implement `BaseStorage` interface:
```python
class RedisStorage(BaseStorage):
    def get(self, key: str) -> Any: ...
    def get_all_keys(self) -> list[str]: ...
    def set(self, key: str, value: Any) -> None: ...
```

### Custom Embedder
Implement `BaseEmbedder` interface or use any LangChain `Embeddings` class.


## License
MIT License
