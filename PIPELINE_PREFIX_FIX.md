# Pipeline Prefix Filtering Fix

## Problem

When running the pipeline with a new PDF, it was processing **all** PDFs in the data directories, not just the specified one.

**Example:**
```bash
python src/run_pipeline.py --pdf data/raw/ASICS-AW23-Run-Catalog.pdf
```

This would process:
- ✗ 12 chunk files (6 ASICS + 6 fy10syb) instead of 6
- ✗ 36 indexes (18 ASICS + 18 fy10syb) instead of 18
- ✗ 432 QA queries (216 ASICS + 216 fy10syb) instead of 216
- ✗ Best config selected from wrong PDF

## Root Cause

The scripts used `--chunk-dir` and processed **ALL** `.jsonl` files found, with no mechanism to filter by specific PDF.

```python
# OLD CODE (processed everything)
chunk_files = sorted(chunk_dir.glob("*.jsonl"))
```

## Solution

Added `--prefix` filtering to 3 critical scripts:

### 1. `build_index.py` (Step 3)
```python
def build_index_grid(..., prefix: str = None):
    chunk_files = sorted(chunk_dir.glob("*.jsonl"))
    chunk_files = [f for f in chunk_files if "__parser_" in f.name]
    
    # NEW: Filter by prefix
    if prefix:
        chunk_files = [f for f in chunk_files if f.stem.startswith(prefix)]
```

### 2. `gen_synth_qa.py` (Step 4)
```python
def generate_qa_grid(..., prefix: str = None):
    chunk_files = sorted(chunk_dir.glob("*.jsonl"))
    
    # NEW: Filter by prefix
    if prefix:
        chunk_files = [f for f in chunk_files if f.stem.startswith(prefix)]
```

### 3. `eval_retrieval.py` (Step 5)
```python
def evaluate_all_indexes(..., prefix: str = None):
    index_dirs = sorted([d for d in indexes_dir.iterdir() 
                        if d.is_dir() and (d / "index.faiss").exists()])
    
    # NEW: Filter by prefix
    if prefix:
        index_dirs = [d for d in index_dirs if d.name.startswith(prefix)]
```

### 4. `run_pipeline.py` - Pass prefix to all steps

The pipeline now passes `self.pdf_path.stem` (e.g., "ASICS-AW23-Run-Catalog") to each script:

```python
# Step 3: Build indexes
cmd = [
    'python', 'src/build_index.py',
    '--chunk-dir', 'data/processed',
    '--output-dir', 'indexes/faiss',
    '--prefix', self.pdf_path.stem  # NEW!
]

# Step 4: Generate QA
cmd = [
    'python', 'src/gen_synth_qa.py',
    '--chunk-dir', 'data/processed',
    '--output-dir', 'data/qa',
    '--model', self.llm_model,
    '--prefix', self.pdf_path.stem  # NEW!
]

# Step 5: Evaluate retrieval
cmd = [
    'python', 'src/eval_retrieval.py',
    '--indexes-dir', 'indexes/faiss',
    '--qa-dir', 'data/qa',
    '--output-dir', 'runs/retrieval',
    '--k-values', '1', '3', '5', '10',
    '--prefix', self.pdf_path.stem  # NEW!
]
```

## Verification

Test the fix:

```bash
# Should now process ONLY ASICS files (6 chunks × 3 embeddings = 18 indexes)
python src/run_pipeline.py --pdf data/raw/ASICS-AW23-Run-Catalog.pdf --start-step 3
```

Expected output:
```
Found 6 chunk files          # ✓ Only ASICS chunks
Using 3 embedding models
Total indexes to build: 18    # ✓ Only ASICS indexes
```

## Benefits

1. ✅ **Isolation**: Each PDF run is isolated from others
2. ✅ **Performance**: Faster execution (no wasted processing)
3. ✅ **Accuracy**: Best config selected from correct PDF
4. ✅ **Scalability**: Can process multiple PDFs in same directories
5. ✅ **Backwards Compatible**: Scripts work without `--prefix` (processes all files)

## Files Modified

- `src/build_index.py` - Added `--prefix` argument and filtering
- `src/gen_synth_qa.py` - Added `--prefix` argument and filtering  
- `src/eval_retrieval.py` - Added `--prefix` argument and filtering
- `src/run_pipeline.py` - Updated Steps 3, 4, 5 to pass `--prefix`
