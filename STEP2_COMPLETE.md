# Step 2 Complete: Parse & Chunk PDFs ✓

## Summary

We successfully implemented **Step 2** of the RAG evaluation pipeline: PDF parsing and chunking with multiple strategies.

## What We Built

### 1. **`src/parse_chunk.py`** - Multi-Parser PDF Processing Module

Implemented three different PDF parsing strategies:

- **PyMuPDF (fitz)**: Fast, basic text extraction with simple heading detection
- **pdfplumber**: Excellent for tables and detailed layout analysis  
- **unstructured**: Layout-aware with semantic blocks (requires additional setup)

Key features:
- Token-based chunking using `tiktoken` (cl100k_base for GPT-4 compatibility)
- Configurable chunk sizes and overlaps
- Semantic block preservation (respects paragraphs, headings, tables)
- Validation to prevent infinite loops (overlap must be < chunk_size)

### 2. **Updated `src/cli.py`** - Command-Line Interface

Added two new commands:

```bash
# Single file chunking
python -m src.cli chunk --doc data/raw/fy10syb.pdf --parser pymupdf --chunk-size 256 --overlap 64

# Grid-based chunking (all configurations from YAML)
python -m src.cli chunk-grid --config configs/grid.default.yaml
```

Features:
- Beautiful Rich terminal output with progress bars
- Support for both matrix (all combinations) and paired configurations
- Comprehensive results table with timing and chunk counts
- Parser comparison analysis

### 3. **`configs/grid.default.yaml`** - Paired Configuration Strategy

Instead of testing all permutations (which would be 3×3=9 combos per parser), we use **paired configurations**:

```yaml
chunk_overlap_pairs:
  - chunk_size: 128, overlap: 32   # 25% overlap
  - chunk_size: 256, overlap: 64   # 25% overlap  
  - chunk_size: 512, overlap: 128  # 25% overlap
```

**Benefits:**
- Meaningful comparisons (consistent 25% overlap ratio)
- Avoids invalid combinations (overlap ≥ chunk_size)
- Reduces total runs: 3 pairs × 2 parsers = **6 combinations** (vs 18 with matrix)

## Results Summary

Successfully processed `fy10syb.pdf` with 6 configurations:

| Parser     | Chunk Size | Overlap | Chunks | Time (s) | Key Insight |
|------------|------------|---------|--------|----------|-------------|
| pymupdf    | 128        | 32      | 689    | 0.24     | Fast, many small chunks |
| pymupdf    | 256        | 64      | 343    | 0.15     | Balanced |
| pymupdf    | 512        | 128     | 199    | 0.15     | Fewer, larger chunks |
| pdfplumber | 128        | 32      | 850    | 3.68     | **23% more chunks** than PyMuPDF |
| pdfplumber | 256        | 64      | 433    | 3.93     | **26% more chunks** than PyMuPDF |
| pdfplumber | 512        | 128     | 250    | 3.96     | **26% more chunks** than PyMuPDF |

### Key Observations:

1. **pdfplumber extracts more content** (+23-26% chunks) - better at capturing tables and structured data
2. **pymupdf is 15-20× faster** but may miss layout details
3. **Chunk count inversely proportional to chunk size** (as expected)
4. **Total chunks created: 2,764** across all configurations

## Files Generated

```
data/processed/
├── fy10syb__parser_pymupdf__cs128__ov32.jsonl      (689 chunks)
├── fy10syb__parser_pymupdf__cs256__ov64.jsonl      (343 chunks)
├── fy10syb__parser_pymupdf__cs512__ov128.jsonl     (199 chunks)
├── fy10syb__parser_pdfplumber__cs128__ov32.jsonl   (850 chunks)
├── fy10syb__parser_pdfplumber__cs256__ov64.jsonl   (433 chunks)
└── fy10syb__parser_pdfplumber__cs512__ov128.jsonl  (250 chunks)
```

Each file contains JSONL with:
```json
{
  "id": "fy10syb__parserpymupdf__chunk0000",
  "text": "U.S. Department of Justice...",
  "token_count": 247,
  "source_blocks": [0],
  "metadata": {"parser": "pymupdf", "chunk_num": 0}
}
```

## What We Learned

1. **Parser choice matters significantly** - pdfplumber's layout analysis captures ~25% more semantic units
2. **Chunk size affects granularity** - smaller chunks = more pieces, better for precise retrieval but more redundancy
3. **Paired configurations** are more practical than full matrix testing
4. **Token-based chunking** ensures consistent LLM context usage

## Next Steps (Step 3): Build Indexes

Now that we have 6 chunked files, we need to:

1. Embed each chunk with multiple embedding models:
   - OpenAI: `text-embedding-3-small`
   - SentenceTransformers: `all-MiniLM-L6-v2`
   - SentenceTransformers: `instructor-large`

2. Build FAISS indexes for each configuration:
   - 6 chunk configs × 3 embeddings = **18 vector indexes**

3. Store indexes in: `indexes/faiss/{doc}__parser_{parser}__cs{cs}__ov{ov}__emb{model}/`

Ready to implement `src/build_index.py`?

---

## Quick Commands Reference

```bash
# Test single chunk
python -m src.cli chunk --doc data/raw/fy10syb.pdf --parser pymupdf --chunk-size 256 --overlap 64

# Run full grid
python -m src.cli chunk-grid --config configs/grid.default.yaml

# List generated chunks
ls -lh data/processed/

# Inspect a chunk file
head -n 1 data/processed/fy10syb__parser_pymupdf__cs256__ov64.jsonl | jq .
```
