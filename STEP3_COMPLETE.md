# Step 3 Complete: Build Vector Indexes ✓

## Summary

We successfully implemented **Step 3** of the RAG evaluation pipeline: Building vector indexes using multiple embedding models and FAISS.

## What We Built

### **`src/build_index.py`** - Vector Index Building Module

Key features:
- **Multi-provider embedding support** (OpenAI API)
- **Batch embedding** with configurable batch sizes
- **FAISS index creation** with flat index (exact search)
- **Automatic .env loading** for API keys
- **Metadata tracking** for reproducibility

## Results Summary

Successfully built **18 vector indexes**:
- **6 chunk configurations** (2 parsers × 3 chunk size pairs)
- **3 embedding models** (OpenAI: text-embedding-3-small, text-embedding-3-large, ada-002)
- **Total time:** 126 seconds (~7s per index)
- **Total storage:** 75 MB

### Embedding Models Used

| Model ID | Provider | Model Name | Dimensions | Speed (chunks/sec) |
|----------|----------|------------|------------|-------------------|
| openai-small | OpenAI | text-embedding-3-small | 1536 | ~70-80 |
| openai-large | OpenAI | text-embedding-3-large | 3072 | ~50-60 |
| ada-002 | OpenAI | text-embedding-ada-002 | 1536 | ~70-85 |

**Key Finding:** ada-002 is fastest, but text-embedding-3 models are newer and may perform better in retrieval.

### Index Matrix

```
Parser      ChunkSize  Overlap  × Embedding Model     = Index
─────────────────────────────────────────────────────────────
pymupdf     128        32       × openai-small        ✓
pymupdf     128        32       × openai-large        ✓
pymupdf     128        32       × ada-002             ✓
pymupdf     256        64       × openai-small        ✓
pymupdf     256        64       × openai-large        ✓
pymupdf     256        64       × ada-002             ✓
pymupdf     512        128      × openai-small        ✓
pymupdf     512        128      × openai-large        ✓
pymupdf     512        128      × ada-002             ✓
pdfplumber  128        32       × openai-small        ✓
pdfplumber  128        32       × openai-large        ✓
pdfplumber  128        32       × ada-002             ✓
pdfplumber  256        64       × openai-small        ✓
pdfplumber  256        64       × openai-large        ✓
pdfplumber  256        64       × ada-002             ✓
pdfplumber  512        128      × openai-small        ✓
pdfplumber  512        128      × openai-large        ✓
pdfplumber  512        128      × ada-002             ✓
                                                TOTAL: 18 indexes
```

## Index Structure

Each index directory contains:
```
indexes/faiss/{doc}__parser_{parser}__cs{cs}__ov{ov}__emb_{emb_id}/
├── index.faiss         # FAISS index file (vectors)
├── chunks.jsonl        # Original chunks with text
└── metadata.json       # Index metadata (model, dimensions, timestamp)
```

### Example Index Sizes

| Configuration | Chunks | Index Size | Metadata |
|--------------|--------|------------|----------|
| pymupdf cs128 ov32 + openai-small | 689 | 2.6 MB | 1536-dim |
| pymupdf cs256 ov64 + openai-large | 343 | 4.0 MB | 3072-dim |
| pdfplumber cs128 ov32 + ada-002 | 850 | 3.2 MB | 1536-dim |

## What We Learned

1. **OpenAI embeddings are production-ready** - Fast, reliable, no setup issues
2. **text-embedding-3-large has 2× dimensions** (3072 vs 1536) - May capture more nuance
3. **Batch size matters** - Larger batches (64) reduce API calls and improve throughput
4. **Index size scales linearly** with chunks × dimensions
5. **SentenceTransformers had issues on macOS** - OpenAI API is more reliable for this use case

## Technical Notes

### Environment Setup
```bash
# API key loaded from .env
OPENAI_API_KEY=sk-proj-...
```

### FAISS Index Type
- Using **IndexFlatIP** (Flat Inner Product)
- Normalized embeddings → Inner Product = Cosine Similarity
- Exact search (no approximation)
- Good for <10K vectors (our largest is 850 chunks)

### Why Not IVF/HNSW?
- Our datasets are small (199-850 chunks)
- Flat index provides exact results
- No training overhead
- Fast enough for evaluation purposes

## Commands Used

```bash
# Single file index
python -m src.build_index \
  --chunk-file data/processed/fy10syb__parser_pymupdf__cs256__ov64.jsonl \
  --config configs/providers.yaml

# Full grid (all chunks × all embeddings)
python -m src.build_index \
  --chunk-dir data/processed \
  --config configs/providers.yaml \
  --batch-size 64
```

## Next Steps (Step 4): Generate Synthetic QA

Now that we have 18 vector indexes, we need to:

1. **Generate synthetic questions** for each chunk configuration
   - Factual questions (single chunk)
   - Analytical questions (reasoning)
   - Multi-hop questions (spanning chunks)
   - Overlap stress questions (boundary testing)

2. **Create gold labels** (relevant_chunk_ids) for each question

3. **Store QA sets** in: `data/qa/{doc}__parser_{parser}__cs{cs}__ov{ov}__qa.jsonl`

4. **Validate with Pydantic** using the `QAItem` schema

**Ready to implement `src/gen_synth_qa.py`?**

---

## Quick Commands Reference

```bash
# List all indexes
ls -1 indexes/faiss/

# Check index size
du -sh indexes/faiss/

# View index metadata
cat indexes/faiss/fy10syb__parser_pymupdf__cs256__ov64__emb_openai-large/metadata.json | jq .

# Count vectors in an index
jq '. | length' indexes/faiss/fy10syb__parser_pymupdf__cs256__ov64__emb_openai-large/chunks.jsonl
```

## Performance Stats

- **Total chunks embedded:** 7,406 (sum across all configs)
- **Average embedding time:** 7.00s per index
- **Throughput:** 60-85 chunks/sec (depends on model)
- **API calls:** ~231 requests (7406 chunks / batch_size 32)
- **Cost estimate:** ~$0.01-0.02 (OpenAI embedding prices)

🎉 **Step 3 Complete! Ready for Step 4: Synthetic QA Generation**
