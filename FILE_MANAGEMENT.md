# File Management: Will My Files Be Overwritten?

## Quick Answer

**Some files will be overwritten, others will be kept.** Here's the breakdown:

## File Handling by Step

### ✅ SAFE: These files are OVERWRITTEN (deterministic names)

These files use the same name every time, so they'll be overwritten on re-run:

| Step | File | Behavior | Reason |
|------|------|----------|---------|
| **Step 2: Parse & Chunk** | `data/processed/{doc}__{parser}__cs{size}__ov{overlap}.jsonl` | **OVERWRITTEN** | Same config = same filename |
| **Step 3: Build Indexes** | `indexes/faiss/{config}__emb_{model}/` | **OVERWRITTEN** | Same config = same directory |
| **Step 4: Generate QA** | `data/qa/{config}__qa.jsonl` | **OVERWRITTEN** | Same config = same filename |
| **Step 5: Eval Retrieval** | `runs/retrieval/retrieval_evaluation_summary.csv` | **OVERWRITTEN** | Single summary file |
| **Step 5: Eval Retrieval** | `runs/retrieval/retrieval_evaluation_detailed.csv` | **OVERWRITTEN** | Single detailed file |
| **Step 7: Generate Answers** | `runs/generation/answers__{config}__emb_{model}__llm_{llm}.jsonl` | **OVERWRITTEN** | Same config = same filename |
| **Step 8: Eval Generation** | `runs/generation/evaluation__{config}__emb_{model}__llm_{llm}.csv` | **OVERWRITTEN** | Derived from answers filename |

### Why This Is Actually Good 🎯

**No File Clutter**: You won't accumulate hundreds of old files  
**Easy Comparison**: Always know where the latest results are  
**Version Control**: Git can track changes between runs  
**Deterministic**: Same input = same output location

## What This Means for You

### Scenario 1: Running the Full Pipeline Again

```bash
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
```

**What happens:**
1. ✅ Chunks will be **regenerated** (same filenames, overwritten)
2. ✅ Indexes will be **rebuilt** (same directories, overwritten)
3. ✅ QA pairs will be **regenerated** (same filenames, overwritten)
4. ✅ Retrieval eval will be **recomputed** (same CSV, overwritten)
5. ✅ Answers will be **regenerated** (same JSONL, overwritten)
6. ✅ LLM-as-Judge eval will be **recomputed** (same CSV, overwritten)

**Result**: All your current results will be **replaced** with new ones.

### Scenario 2: Backing Up Before Re-running

If you want to keep your current results:

```bash
# Option 1: Backup the entire runs directory
cp -r runs runs_backup_$(date +%Y%m%d_%H%M%S)

# Option 2: Backup specific results
cp runs/generation/evaluation__*.csv evaluation_backup.csv
cp runs/generation/answers__*.jsonl answers_backup.jsonl

# Then run pipeline
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
```

### Scenario 3: Testing with Different Documents

If you run with a **different PDF**, files won't conflict:

```bash
# Current files: fy10syb__*
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf

# New files: another_doc__* (no conflict!)
python src/run_pipeline.py --pdf data/raw/another_doc.pdf
```

Each PDF gets its own set of files based on the document name.

### Scenario 4: Testing with Different Configurations

If you manually run individual scripts with **different parameters**:

```bash
# Creates: fy10syb__parser_pdfplumber__cs512__ov128.jsonl
python src/parse_chunk.py --pdf data/raw/fy10syb.pdf --parser pdfplumber \
    --chunk-size 512 --overlap 128

# Creates: fy10syb__parser_pdfplumber__cs256__ov64.jsonl (different file!)
python src/parse_chunk.py --pdf data/raw/fy10syb.pdf --parser pdfplumber \
    --chunk-size 256 --overlap 64
```

Different parameters = different filenames = no overwriting.

## Directory Structure After Re-run

```
rag-pdf-project/
├── data/
│   ├── processed/
│   │   ├── fy10syb__parser_pymupdf__cs128__ov32.jsonl      (overwritten)
│   │   ├── fy10syb__parser_pymupdf__cs256__ov64.jsonl      (overwritten)
│   │   ├── fy10syb__parser_pymupdf__cs512__ov128.jsonl     (overwritten)
│   │   ├── fy10syb__parser_pdfplumber__cs128__ov32.jsonl   (overwritten)
│   │   ├── fy10syb__parser_pdfplumber__cs256__ov64.jsonl   (overwritten)
│   │   └── fy10syb__parser_pdfplumber__cs512__ov128.jsonl  (overwritten)
│   │
│   └── qa/
│       ├── fy10syb__parser_pymupdf__cs128__ov32__qa.jsonl      (overwritten)
│       ├── fy10syb__parser_pymupdf__cs256__ov64__qa.jsonl      (overwritten)
│       └── ... (all QA files overwritten)
│
├── indexes/
│   └── faiss/
│       ├── fy10syb__parser_pymupdf__cs128__ov32__emb_ada-002/  (overwritten)
│       ├── fy10syb__parser_pymupdf__cs128__ov32__emb_text-embedding-3-small/  (overwritten)
│       └── ... (all 18 index directories overwritten)
│
└── runs/
    ├── retrieval/
    │   ├── retrieval_evaluation_summary.csv    (overwritten)
    │   └── retrieval_evaluation_detailed.csv   (overwritten)
    │
    └── generation/
        ├── answers__fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002__llm_gpt-4o-mini.jsonl  (overwritten)
        └── evaluation__fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002__llm_gpt-4o-mini.csv  (overwritten)
```

## Best Practices

### ✅ DO: Commit to Git Before Re-running

```bash
# Save your current results
git add -A
git commit -m "Results: fy10syb evaluation - 91.7% recall, 4.4/5.0 quality"

# Now safe to re-run
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
```

Git tracks changes, so you can always go back:
```bash
# See what changed
git diff

# Restore old results if needed
git checkout HEAD~1 runs/
```

### ✅ DO: Create Timestamped Backups for Important Runs

```bash
# Before a production evaluation
tar -czf results_$(date +%Y%m%d_%H%M%S).tar.gz runs/ data/qa/

# Or just copy
cp -r runs runs_production_backup
```

### ✅ DO: Use Different PDFs for Testing

```bash
# Test on smaller/different document
python src/run_pipeline.py --pdf data/raw/test_doc.pdf --quick
```

This creates `test_doc__*` files that won't interfere with `fy10syb__*`.

### ❌ DON'T: Worry About Intermediate Files

The pipeline is designed to be **idempotent** (safe to re-run):
- Chunks regenerate quickly (~15 seconds)
- Indexes rebuild efficiently (~2 minutes)
- Old results are replaced with fresh ones

## Why Not Use Timestamps in Every Filename?

You might ask: "Why not append timestamps to every file to avoid overwriting?"

**Answer**: Deliberate design decision for clarity:

### Problems with Timestamps Everywhere

```
# Nightmare: Which is the current result?
runs/generation/answers__20251020-133933.jsonl
runs/generation/answers__20251020-141522.jsonl
runs/generation/answers__20251020-153044.jsonl
runs/generation/answers__20251020-162311.jsonl
runs/generation/answers__20251020-175629.jsonl

# You'd need to:
# 1. Sort by timestamp
# 2. Pick the latest
# 3. Hope it corresponds to the right config
# 4. Clean up old files manually
```

### Benefits of Deterministic Names

```
# Clear: This is the result for this exact config
runs/generation/answers__fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002__llm_gpt-4o-mini.jsonl

# Easy to:
# 1. Find the current result immediately
# 2. Know exactly which config it corresponds to
# 3. Compare before/after with git diff
# 4. No manual cleanup needed
```

## Advanced: Custom Backup Script

Create a script to auto-backup before runs:

```bash
#!/bin/bash
# save_results.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$TIMESTAMP"

echo "Backing up current results to $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"

cp -r runs/ "$BACKUP_DIR/"
cp -r data/qa/ "$BACKUP_DIR/"

echo "✓ Backup complete: $BACKUP_DIR"
echo "You can now safely re-run the pipeline!"
```

Usage:
```bash
./save_results.sh
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
```

## Summary

| If you want to... | Do this... |
|-------------------|------------|
| **Keep current results** | Backup `runs/` before re-running: `cp -r runs runs_backup` |
| **Compare old vs new** | Use git: `git add -A && git commit -m "Current results"` |
| **Test without overwriting** | Use a different PDF: `--pdf data/raw/test.pdf` |
| **Just re-run safely** | Go ahead! Results are reproducible and can be regenerated |

## The Bottom Line

**Files WILL be overwritten**, but this is by design for:
- 🎯 **Clarity**: Always know where current results are
- 🔄 **Reproducibility**: Same inputs → same outputs → same locations  
- 🧹 **Cleanliness**: No file clutter
- 📊 **Version Control**: Git tracks changes

**Recommendation**: Commit your current results to git, then re-run with confidence! Your current excellent results (91.7% recall, 4.4/5.0 quality) are preserved in git history.

```bash
# Safe workflow:
git add -A
git commit -m "Baseline: 91.7% recall, 4.4/5.0 quality, pdfplumber+cs512"
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
```

You're now free to experiment! 🚀
