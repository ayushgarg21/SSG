# Performance Optimization Guide

## Performance Comparison

### 1. **Batch Processing** ([jobert_batch.py](jobert_batch.py))
**Best for: Most use cases, especially with GPU**

**Pros:**
- 🚀 **5-10x faster** than single-sentence inference
- Works with GPU (automatically uses CUDA if available)
- Simplest code changes
- No multiprocessing overhead

**Cons:**
- Still processes rows sequentially
- Not ideal for very large datasets (>10k rows)

**When to use:**
- You have a GPU
- Dataset < 10k rows
- You want the simplest speedup

---

### 2. **Multiprocessing** ([jobert_multiprocessing.py](jobert_multiprocessing.py))
**Best for: Large CPU-only datasets**

**Pros:**
- Processes multiple rows in parallel
- Good for CPU-only environments
- Works well with joblib
- Combines well with batch processing

**Cons:**
- Model loaded multiple times (once per worker = high memory)
- Slower startup time
- May not work well with GPU (model copying issues)
- Overhead for small datasets

**When to use:**
- No GPU available
- Dataset > 1k rows
- You have multiple CPU cores (4+)
- Memory is not a constraint

**Setup:**
```bash
pip install joblib
```

---

### 3. **Dask** ([jobert_dask.py](jobert_dask.py))
**Best for: Very large datasets (>100k rows) or out-of-memory data**

**Pros:**
- Handles datasets larger than RAM
- Good for distributed computing
- Familiar pandas-like API
- Can scale to clusters

**Cons:**
- Overhead for small datasets
- More complex setup
- Model loading per partition can be slow
- Not ideal with GPU

**When to use:**
- Dataset > 100k rows
- Data doesn't fit in memory
- You need to scale to multiple machines
- You're already using Dask elsewhere

**Setup:**
```bash
pip install dask[complete]
```

---

---

## 🚀 FOR 16M ROWS: USE MEGA-BATCH PROCESSING

### **[jobert_megabatch_mps.py](jobert_megabatch_mps.py)** - RECOMMENDED FOR LARGE DATASETS

**Key innovation:** Collects sentences from multiple rows, classifies in mega-batches (256+), then distributes results back.

**Pros:**
- **10-20x faster** than row-by-row processing
- Works with MPS (Apple Silicon), CUDA, or CPU
- Memory efficient (processes in chunks)
- Saves intermediate results (resume if interrupted)
- Handles datasets larger than RAM

**Strategy for 16M rows:**
1. Reads data in chunks (5k rows at a time)
2. Extracts all sentences from chunk
3. Classifies ALL sentences in mega-batches on GPU
4. Distributes results back to rows
5. Saves every 50k rows (resumable)

**Expected time for 16M rows:**
- With MPS/GPU: ~8-12 hours
- With CPU: ~2-3 days

**Memory usage:** ~4-8GB (processes in chunks)

---

### **[jobert_dask_mps.py](jobert_dask_mps.py)** - Alternative for MPS

Uses Dask for partitioning but avoids multiprocessing (MPS limitation).
Good if you need Dask's features, but mega-batch is faster.

---

## Recommended Approach

### **Start with Batch Processing** ([jobert_batch.py](jobert_batch.py))

For small datasets (100-1k rows), batch processing will give you the biggest speedup with minimal code changes:

1. **Key change:** Process multiple sentences at once
   ```python
   # Old: Process one at a time
   for sentence in sentences:
       label = classify_sentence(sentence, ...)

   # New: Process all at once
   labels = classify_sentences_batch(sentences, batch_size=32)
   ```

2. **Expected speedup:** 5-10x faster inference
3. **No dependencies needed:** Works with existing code

### **If you need more speed:**

**For 1k-10k rows with CPU:**
→ Use [jobert_multiprocessing.py](jobert_multiprocessing.py) (multiprocessing + batch processing)

**For 100k+ rows:**
→ Use [jobert_dask.py](jobert_dask.py)

---

## Performance Tips

### Additional Optimizations

1. **Use GPU if available:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   ```

2. **Tune batch size:**
   - GPU: Try 64-128
   - CPU: Try 16-32
   - Larger = faster but more memory

3. **Disable spaCy pipes you don't need:**
   ```python
   nlp = en_core_web_lg.load()
   nlp.disable_pipes(['ner', 'parser'])  # Keep only tokenizer
   ```

4. **Use smaller spaCy model:**
   ```python
   # Instead of en_core_web_lg (large)
   nlp = en_core_web_sm.load()  # small - 3x faster
   ```

5. **Process in chunks:**
   ```python
   # For very large datasets, process in chunks
   for chunk in pd.read_parquet(file_path, chunksize=1000):
       process(chunk)
   ```

---

## Quick Decision Guide

| Dataset Size | Hardware | Use This | Expected Time |
|-------------|----------|----------|---------------|
| < 1k rows | Any | [jobert_batch.py](jobert_batch.py) | Minutes |
| 1k-10k rows | CPU only | [jobert_multiprocessing.py](jobert_multiprocessing.py) | 10-30 min |
| 1k-10k rows | GPU/MPS | [jobert_batch.py](jobert_batch.py) | 5-15 min |
| 10k-100k rows | GPU/MPS | [jobert_megabatch_mps.py](jobert_megabatch_mps.py) | 30-90 min |
| 100k-16M rows | GPU/MPS | [jobert_megabatch_mps.py](jobert_megabatch_mps.py) | 4-12 hours |
| 100k-16M rows | CPU only | [jobert_megabatch_mps.py](jobert_megabatch_mps.py) | 1-3 days |

## Quick Benchmark (Expected Times)

### For 100 rows:
| Method | Time (approx) | Speedup |
|--------|---------------|---------|
| Original (single-sentence) | ~10 min | 1x |
| Batch processing (CPU) | ~2 min | 5x |
| Batch processing (GPU) | ~30 sec | 20x |
| Multiprocessing (4 cores) | ~3 min | 3-4x |

### For 16M rows:
| Method | MPS/GPU | CPU |
|--------|---------|-----|
| Original | ~30 days 💀 | ~60 days 💀 |
| Batch | ~3 days | ~7 days |
| **Mega-batch** | **~8-12 hours** ✅ | **~2-3 days** |

*Note: Actual times depend on your hardware and sentence counts*
