# Step 9: End-to-End Pipeline Runner

## Overview

The **End-to-End Pipeline Runner** (`src/run_pipeline.py`) orchestrates all 9 steps of the RAG evaluation pipeline in a single command. This is the culmination of our systematic evaluation approach!

## What It Does

The pipeline runner automatically executes:

1. **Step 1**: Load & Verify PDF
2. **Step 2**: Parse & Chunk (multiple configurations)
3. **Step 3**: Build Vector Indexes (FAISS with multiple embeddings)
4. **Step 4**: Generate Synthetic QA pairs
5. **Step 5**: Evaluate Retrieval (Recall@K, Precision@K, MRR@K)
6. **Step 6**: Reranking (skipped if retrieval excellent)
7. **Step 7**: Generate Answers (RAG with citations)
8. **Step 8**: Evaluate Generation (LLM-as-Judge)
9. **Step 9**: Final Analysis & Recommendations

## Usage

### Full Evaluation Mode (Recommended)

Evaluates all configurations to find the absolute best setup:

```bash
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
```

**Configurations tested**:
- 2 parsers (PyMuPDF, pdfplumber)
- 3 chunk sizes (128, 256, 512 tokens)
- 3 overlaps (32, 64, 128 tokens)
- 3 embeddings (text-embedding-3-small, text-embedding-3-large, ada-002)
- **Total**: 18 configurations

**Time**: ~15-20 minutes  
**Cost**: ~$0.10 (API calls)

### Quick Mode (For Testing)

Tests a single configuration for rapid iteration:

```bash
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf --quick
```

**Configurations tested**:
- 1 parser (pdfplumber)
- 1 chunk size (512 tokens, 128 overlap)
- 1 embedding (ada-002)
- **Total**: 1 configuration

**Time**: ~5 minutes  
**Cost**: ~$0.02

### Skip Evaluation Mode

Generates chunks and indexes without running expensive evaluations:

```bash
python src/run_pipeline.py --pdf data/raw/fy10syb.pdf --skip-eval
```

**Use cases**:
- Just want to generate chunks/indexes
- Testing infrastructure without API costs
- Preparing data for manual evaluation

**Time**: ~2 minutes  
**Cost**: $0 (no API calls)

## Command-Line Options

```
--pdf PATH              Path to PDF file to evaluate (required)
--output-dir PATH       Root output directory (default: current directory)
--quick                 Quick mode: single configuration for testing
--skip-eval             Skip expensive evaluation steps (no API calls)
```

## Output Structure

The pipeline generates a complete directory structure:

```
rag-pdf-project/
├── data/
│   ├── raw/                    # Input PDFs
│   ├── processed/              # Chunked documents (JSONL)
│   └── qa/                     # Synthetic QA pairs
├── indexes/
│   └── faiss/                  # Vector indexes
└── runs/
    ├── retrieval/              # Retrieval evaluation results
    │   ├── retrieval_evaluation_summary.csv
    │   └── retrieval_evaluation_detailed.csv
    └── generation/             # Generation evaluation results
        ├── answers__*.jsonl
        └── evaluation__*.csv
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Document                             │
│                  (fy10syb.pdf)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Load PDF                                           │
│  └─ Verify file exists, check size                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Parse & Chunk                                      │
│  ├─ Parser 1 (PyMuPDF) × 3 chunk configs                    │
│  ├─ Parser 2 (pdfplumber) × 3 chunk configs                 │
│  └─ Output: 6 chunked JSONL files                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Build Vector Indexes                               │
│  ├─ 6 chunk configs × 3 embeddings                          │
│  └─ Output: 18 FAISS indexes (~75MB)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Generate Synthetic QA                              │
│  ├─ 12 questions per config (factual, analytical, etc.)     │
│  └─ Output: 72 QA pairs with gold answers                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Evaluate Retrieval                                 │
│  ├─ Test all 18 configs on QA pairs                         │
│  ├─ Compute Recall@K, Precision@K, MRR@K                    │
│  └─ Find best configuration → pdfplumber+cs512+ada-002      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Reranking (Optional)                               │
│  └─ Skipped (91.7% recall already excellent!)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 7: Generate Answers                                   │
│  ├─ Use best config from Step 5                             │
│  ├─ RAG with structured JSON output                         │
│  └─ Output: 12 answers with citations                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 8: Evaluate Generation (LLM-as-Judge)                 │
│  ├─ Faithfulness, Relevance, Completeness, Citations        │
│  └─ Output: 4.4/5.0 overall quality                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 9: Final Analysis                                     │
│  ├─ Summary table of all metrics                            │
│  ├─ Best configuration recommendation                       │
│  └─ Next steps for deployment                               │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 🎯 Smart Configuration Management

The runner automatically:
- Discovers all chunking configurations
- Builds indexes for all embedding models
- Identifies the best performing configuration
- Uses the best config for answer generation

### 📊 Progress Tracking

Beautiful terminal output with:
- Step-by-step progress indicators
- Real-time status updates
- Detailed error messages
- Summary tables and metrics

### 🛡️ Error Handling

Robust error handling:
- Validates inputs before starting
- Stops on critical errors
- Logs all errors for debugging
- Provides clear error messages

### ⚡ Performance Modes

Three modes for different use cases:
1. **Full**: Comprehensive evaluation (production)
2. **Quick**: Single config testing (development)
3. **Skip-Eval**: Infrastructure only (no API costs)

## Example Run

```bash
$ python src/run_pipeline.py --pdf data/raw/fy10syb.pdf

╔═══════════════════════════════════════════════════════════════╗
║            🔬 FULL EVALUATION                                 ║
║                                                               ║
║  PDF: fy10syb.pdf                                            ║
║  Output: /Users/vb/Software/ai-project/rag-pdf-project      ║
║  Configs: 2 parsers × 3 chunks × 3 embeddings                ║
║  Total: 18 configurations                                     ║
╚═══════════════════════════════════════════════════════════════╝

═══ Step 1: Load PDF ═══
✓ Found PDF: fy10syb.pdf (1.23 MB)

═══ Step 2: Parse & Chunk ═══
Generating 6 chunked versions...
  Parsing & chunking... ━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:15
✓ Created 6 chunked configurations

═══ Step 3: Build Vector Indexes ═══
Building 18 FAISS indexes...
  Building indexes... ━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:06
✓ Built 18 vector indexes

═══ Step 4: Generate Synthetic QA ═══
  Generating questions... ━━━━━━━━━━━━━━━━━━ 100% 0:05:12
✓ Generated synthetic QA pairs

═══ Step 5: Evaluate Retrieval ═══
  Evaluating retrieval... ━━━━━━━━━━━━━━━━━━ 100% 0:03:24

🏆 Best Configuration:
  Config: fy10syb__parser_pdfplumber__cs512__ov128
  Embedding: ada-002
  Recall@5: 91.7%

✓ Evaluated retrieval performance

═══ Step 6: Reranking ═══
⊘ Skipped (retrieval already excellent)

═══ Step 7: Generate Answers ═══
  Generating answers... ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:35
✓ Generated answers with citations

═══ Step 8: Evaluate Generation (LLM-as-Judge) ═══
  Evaluating answers... ━━━━━━━━━━━━━━━━━━━━ 100% 0:00:29
✓ Evaluated generation quality

═══ Step 9: Final Analysis ═══

                    Pipeline Summary                    
┌────────────────────────┬──────────────────────────────┐
│ Metric                 │ Value                        │
├────────────────────────┼──────────────────────────────┤
│ PDF Document           │ fy10syb.pdf                  │
│ Configurations Tested  │ 18                           │
│ Steps Completed        │ 9                            │
│ Steps Skipped          │ 1                            │
│ Errors                 │ 0                            │
│ Total Time             │ 15.3 minutes                 │
│ Best Config            │ pdfplumber__cs512__ov128     │
│ Best Embedding         │ ada-002                      │
│ Best Recall@5          │ 91.7%                        │
└────────────────────────┴──────────────────────────────┘

✅ Pipeline Complete!

Next Steps:
  1. Review results in runs/generation/ and runs/retrieval/
  2. Check evaluation CSVs for detailed metrics
  3. Read STEP8_COMPLETE.md for quality analysis
  4. Test on additional documents
  5. Deploy to production!
```

## Integration with Existing Tools

The pipeline runner orchestrates individual scripts:

| Step | Script | Purpose |
|------|--------|---------|
| 2 | `src/parse_chunk.py` | Parse & chunk PDFs |
| 3 | `src/build_index.py` | Build FAISS indexes |
| 4 | `src/gen_synth_qa.py` | Generate QA pairs |
| 5 | `src/eval_retrieval.py` | Evaluate retrieval |
| 7 | `src/generate_answers.py` | Generate answers |
| 8 | `src/eval_generation.py` | LLM-as-Judge |

You can still run individual scripts for debugging or custom workflows!

## Advanced Usage

### Custom Configuration

Create a custom YAML config:

```yaml
# configs/custom.yaml
parsers:
  - pdfplumber
chunk_configs:
  - {size: 384, overlap: 96}
  - {size: 768, overlap: 192}
embeddings:
  - text-embedding-3-large
llm: gpt-4o-mini
k: 10
```

Then run:

```bash
python src/run_pipeline.py --pdf data/raw/doc.pdf --config configs/custom.yaml
```

### Multiple Documents

Evaluate multiple PDFs in batch:

```bash
for pdf in data/raw/*.pdf; do
    python src/run_pipeline.py --pdf "$pdf" --quick
done
```

### Continue from Failure

If the pipeline fails at a step, you can:

1. Fix the issue
2. Run individual step scripts manually
3. Continue with remaining steps

Example:

```bash
# Pipeline failed at Step 5
# Fix the issue, then run remaining steps manually:
python src/generate_answers.py --config best_config --embedding ada-002
python src/eval_generation.py --answers-file runs/generation/answers__*.jsonl
```

## Performance Optimization

### Parallel Processing

For multiple documents, run in parallel:

```bash
# Using GNU parallel
parallel -j 4 python src/run_pipeline.py --pdf {} --quick ::: data/raw/*.pdf
```

### Resource Management

- **Memory**: FAISS indexes use ~75MB total
- **Disk**: Chunked files + indexes ~100MB per document
- **API**: Full evaluation ~$0.10 per document

### Caching

The pipeline caches intermediate results:
- Chunked documents in `data/processed/`
- Vector indexes in `indexes/faiss/`
- QA pairs in `data/qa/`

Subsequent runs reuse cached data if inputs haven't changed.

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'rich'`
```bash
pip install -r requirements.txt
```

**Issue**: `OpenAI API key not found`
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Issue**: `PDF file not found`
```bash
# Check file path is correct
ls -lh data/raw/
```

**Issue**: Pipeline hangs at indexing step
```bash
# Check available memory
# FAISS requires ~5GB RAM for large documents
```

### Debug Mode

Run individual steps with verbose output:

```bash
python src/parse_chunk.py --pdf data/raw/doc.pdf --parser pdfplumber \
    --chunk-size 512 --overlap 128 --verbose
```

## What You Learned

### Key Concepts

1. **Pipeline Orchestration**: Chaining multiple steps into cohesive workflow
2. **Configuration Management**: Systematic testing of parameter combinations
3. **Error Handling**: Robust failure detection and recovery
4. **Progress Tracking**: User-friendly status updates
5. **Result Caching**: Efficient reuse of intermediate outputs

### Design Patterns

- **Builder Pattern**: Constructing complex pipeline configurations
- **Chain of Responsibility**: Each step depends on previous step's output
- **Template Method**: Common structure with customizable steps
- **Strategy Pattern**: Different modes (full/quick/skip-eval)

### Best Practices

✅ **Validate inputs early** - Fail fast on missing files or invalid configs  
✅ **Log everything** - Comprehensive error messages and progress tracking  
✅ **Cache intermediate results** - Don't recompute expensive operations  
✅ **Provide escape hatches** - Allow manual intervention at any step  
✅ **Make it resumable** - Can continue from failure point  

## Next Steps

Now that you have the end-to-end runner:

1. **Test on your own documents** - Try different PDF types
2. **Tune configurations** - Experiment with chunk sizes and embeddings
3. **Add custom steps** - Extend the pipeline with domain-specific logic
4. **Deploy to production** - Use the best config in your application
5. **Monitor in production** - Track quality metrics over time

## Summary

The End-to-End Pipeline Runner is the **culmination of your RAG evaluation project**. It:

- ✅ Automates all 9 steps
- ✅ Tests multiple configurations systematically
- ✅ Finds the optimal setup automatically
- ✅ Provides production-ready outputs
- ✅ Costs only ~$0.10 and 15 minutes per document

**You now have a complete, systematic RAG evaluation framework!** 🎉

---

**Files**:
- **Runner**: `src/run_pipeline.py` (500+ lines)
- **Documentation**: `STEP9_COMPLETE.md` (this file)
- **Usage**: `python src/run_pipeline.py --pdf data/raw/fy10syb.pdf`
