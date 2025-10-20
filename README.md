# RAG Evaluation Pipeline

> A systematic framework for evaluating and optimizing Retrieval-Augmented Generation (RAG) systems for PDF documents.

## 🎯 Project Goal

Build a reproducible pipeline to systematically evaluate RAG configurations:
1. **Parse & Chunk** PDFs with multiple strategies
2. **Index** with various embedding models
3. **Generate** synthetic QA pairs with gold labels
4. **Evaluate** retrieval performance (Recall@K, Precision@K, MRR@K)
5. **Assess** generation quality (faithfulness, relevance, completeness)
6. **Identify** optimal configuration for your documents

## 🚀 Quick Start

```bash
# Run the complete pipeline
python src/run_pipeline.py --pdf data/raw/your_document.pdf

# Quick mode (single configuration for testing)
python src/run_pipeline.py --pdf data/raw/your_document.pdf --quick

# Skip evaluation (just build infrastructure)
python src/run_pipeline.py --pdf data/raw/your_document.pdf --skip-eval
```

## 📊 Current Best Configuration

> **📋 See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for complete evaluation results, comparison tables, and detailed analysis**

**Retrieval Performance:**
- **Parser:** pdfplumber (+85% more content vs PyMuPDF)
- **Chunks:** 512 tokens with 128 overlap (25% ratio)
- **Embedding:** text-embedding-ada-002
- **Recall@5:** 91.7% (excellent!)

**Generation Quality:**
- **LLM:** gpt-4o-mini
- **Faithfulness:** 4.5/5.0 (minimal hallucinations)
- **Relevance:** 5.0/5.0 (perfect - always answers question)
- **Completeness:** 4.33/5.0 (comprehensive coverage)
- **Citations:** 3.75/5.0 (good grounding)
- **Overall:** 4.4/5.0 (production-ready!)

## 📁 Project Structure

```
rag-pdf-project/
├── data/
│   ├── raw/                    # Source PDFs
│   ├── processed/              # Chunked documents (JSONL)
│   └── qa/                     # Synthetic QA pairs with gold labels
├── indexes/
│   ├── faiss/                  # FAISS vector indexes
│   ├── lancedb/                # (Optional) LanceDB tables
│   └── dryrun/                 # Dry-run metadata
├── runs/
│   ├── retrieval/              # Retrieval evaluation results (CSV)
│   └── generation/             # Answer generation + LLM-as-Judge (CSV/JSONL)
├── src/
│   ├── config.py               # Configuration schemas (Pydantic)
│   ├── parse_chunk.py          # PDF parsing + token-aware chunking
│   ├── build_index.py          # Vector index building (FAISS)
│   ├── gen_synth_qa.py         # Synthetic QA generation
│   ├── eval_retrieval.py       # Retrieval evaluation (Recall/Precision/MRR)
│   ├── rerank.py               # Reranking (optional)
│   ├── generate_answers.py     # Answer generation with citations
│   ├── eval_generation.py      # LLM-as-Judge evaluation
│   ├── run_pipeline.py         # End-to-end orchestrator
│   └── utils_logging.py        # Logging utilities
├── configs/
│   ├── grid.default.yaml       # Evaluation grid (chunks, embeddings, K)
│   └── providers.yaml          # API provider configuration
├── prompts/
│   ├── retriever_aware_answering.txt
│   └── LLM-as-judge.txt
├── requirements.txt
├── .env                        # API keys (not committed)
└── README.md
```

## 🔧 Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Add Your Documents

Place PDF files in `data/raw/`:

```bash
cp your_document.pdf data/raw/
```

## 📖 Pipeline Steps

### Step 1: Dataset

**Input:** PDF documents in `data/raw/`

**Sources:**
- Your own PDFs
- Kaggle "Enterprise RAG Markdown" dataset
- Any technical documentation

### Step 2: Parsing & Chunking

**Script:** `src/parse_chunk.py`

**What it does:**
- Parses PDFs using PyMuPDF or pdfplumber
- Extracts structured blocks (headings, paragraphs, lists, tables)
- Creates token-aligned chunks with configurable size and overlap
- Preserves document structure and context

**Configuration:**
```yaml
chunk_sizes: [128, 256, 512]    # tokens per chunk
overlaps: [32, 64, 128]         # overlap tokens (25% ratio)
parsers: ["pymupdf", "pdfplumber"]
```

**Output:** `data/processed/{doc}__parser_{parser}__cs{size}__ov{overlap}.jsonl`

**Usage:**
```bash
python src/parse_chunk.py \
    --pdf data/raw/doc.pdf \
    --parser pdfplumber \
    --chunk-size 512 \
    --overlap 128
```

### Step 3: Vector Indexing

**Script:** `src/build_index.py`

**What it does:**
- Batch-embeds chunks with multiple embedding models
- Builds FAISS indexes for efficient similarity search
- Supports multiple embedding providers (OpenAI, Cohere, etc.)

**Configuration:**
```yaml
embeddings:
  - openai:text-embedding-3-small (1536d)
  - openai:text-embedding-3-large (3072d)
  - openai:text-embedding-ada-002 (1536d)
```

**Output:** `indexes/faiss/{config}__emb_{model}/`

**Usage:**
```bash
python src/build_index.py \
    --chunks data/processed/doc__parser_pdfplumber__cs512__ov128.jsonl \
    --embedding text-embedding-ada-002
```

### Step 4: Synthetic QA Generation

**Script:** `src/gen_synth_qa.py`

**What it does:**
- Generates diverse question types from chunks
- Creates gold labels (relevant chunk IDs)
- Stress-tests retrieval with boundary and multi-hop questions

**Question Types:**
1. **Factual:** "What is X?"
2. **Analytical:** "Why does X happen?"
3. **Multi-hop:** Questions requiring multiple chunks
4. **Boundary:** Questions spanning chunk boundaries

**Output:** `data/qa/{config}__qa.jsonl`

**Usage:**
```bash
python src/gen_synth_qa.py \
    --chunks data/processed/doc__parser_pdfplumber__cs512__ov128.jsonl \
    --llm gpt-4o-mini \
    --num-questions 12
```

### Step 5: Retrieval Evaluation

**Script:** `src/eval_retrieval.py`

**What it does:**
- Evaluates retrieval across all configurations
- Computes Recall@K, Precision@K, MRR@K
- Identifies best performing setup

**Metrics:**
- **Recall@K:** Fraction of gold chunks in top-K (primary metric)
- **Precision@K:** Accuracy of retrieved chunks
- **MRR@K:** Mean Reciprocal Rank (ranking quality)

**Output:** `runs/retrieval/retrieval_evaluation_summary.csv`

**Usage:**
```bash
python src/eval_retrieval.py \
    --qa-dir data/qa/ \
    --indexes-dir indexes/faiss/ \
    --k 5
```

### Step 6: Reranking (Optional)

**Script:** `src/rerank.py`

**What it does:**
- Applies reranking to top-M retrieved chunks
- Uses Cohere or open-source models
- Improves precision for challenging queries

**Note:** Skipped in current pipeline due to excellent baseline recall (91.7%)

### Step 7: Answer Generation

**Script:** `src/generate_answers.py`

**What it does:**
- Generates grounded answers using best configuration
- Extracts citations from retrieved chunks
- Uses structured JSON output for reliable parsing

**Output:** `runs/generation/answers__{config}.jsonl`

**Format:**
```json
{
  "question_id": "q_001",
  "question": "What is X?",
  "answer": "X is...",
  "citations": [
    {"chunk_id": "chunk_42", "text": "...relevant excerpt..."}
  ]
}
```

**Usage:**
```bash
python src/generate_answers.py \
    --qa data/qa/doc__qa.jsonl \
    --index indexes/faiss/best_config/ \
    --llm gpt-4o-mini \
    --k 5
```

### Step 8: Generation Evaluation (LLM-as-Judge)

**Script:** `src/eval_generation.py`

**What it does:**
- Evaluates answer quality with LLM judge
- Assesses faithfulness, relevance, completeness, citations
- Provides actionable feedback

**Criteria:**
1. **Faithfulness (0-5):** Grounding in retrieved text
2. **Relevance (0-5):** Addresses the question
3. **Completeness (0-5):** Comprehensive coverage
4. **Citation Quality (0-5):** Proper attribution

**Output:** `runs/generation/evaluation__{config}.csv`

**Usage:**
```bash
python src/eval_generation.py \
    --answers runs/generation/answers__best_config.jsonl \
    --judge-llm gpt-4o-mini
```

### Step 9: End-to-End Pipeline

**Script:** `src/run_pipeline.py`

**What it does:**
- Orchestrates all steps automatically
- Manages dependencies between steps
- Provides progress tracking and summaries

**Modes:**
1. **Full:** Complete evaluation (18 configurations)
2. **Quick:** Single configuration for testing
3. **Skip-eval:** Infrastructure only (no API calls)

## 📊 Evaluation Grid

The pipeline tests all combinations:

```
Parsers (2):        PyMuPDF, pdfplumber
Chunk Sizes (3):    128, 256, 512 tokens
Overlaps (3):       32, 64, 128 tokens (25% ratio)
Embeddings (3):     ada-002, 3-small, 3-large
K Values (1):       5

Total: 2 × 3 × 3 × 3 = 18 configurations
```

## 💰 Cost & Performance

**Full Pipeline (18 configs):**
- Time: ~15 minutes
- Cost: ~$0.10 (OpenAI API)
- Storage: ~75MB (FAISS indexes)

**Quick Mode (1 config):**
- Time: ~5 minutes
- Cost: ~$0.02
- Storage: ~4MB

**Production (per answer):**
- Time: ~3 seconds
- Cost: ~$0.002

## 🎓 Key Learnings

### Parser Selection
- **pdfplumber:** 85% more content extraction (220 blocks vs 119)
- **PyMuPDF:** Faster but misses complex layouts

### Chunk Size Trade-offs
- **Small (128):** Better recall, more redundancy, higher cost
- **Medium (256):** Balanced performance
- **Large (512):** Best semantic cohesion, lower recall

### Overlap Strategy
- **25% ratio** (e.g., 512:128) optimal for context preservation
- Too little: Missing boundary information
- Too much: Redundancy without benefit

### Embedding Models
- **ada-002:** Best cost/performance balance
- **3-small:** Similar performance, slightly cheaper
- **3-large:** Marginal gains, 2x cost

## 🔍 Advanced Usage

### Individual Scripts

Run pipeline steps independently for debugging:

```bash
# Parse only
python src/parse_chunk.py --pdf data/raw/doc.pdf --parser pdfplumber

# Index only
python src/build_index.py --chunks data/processed/doc*.jsonl

# Evaluate specific configuration
python src/eval_retrieval.py --config-filter "cs512__ov128"
```

### Custom Configurations

Edit `configs/grid.default.yaml`:

```yaml
chunk_sizes: [256, 512, 1024]  # Test larger chunks
embeddings:
  - openai:text-embedding-3-large  # Use only premium model
k_values: [3, 5, 10]  # Test different K values
```

### Backup Results

```bash
# Before re-running pipeline
./save_results.sh

# Manual backup
cp -r runs runs_backup_$(date +%Y%m%d_%H%M%S)
```

## 📝 Documentation

Detailed guides available:
- **`RESULTS_SUMMARY.md`** - Complete evaluation results and analysis ⭐
- `STEP4_COMPLETE.md` - Synthetic QA generation
- `STEP5_COMPLETE.md` - Retrieval evaluation
- `STEP7_COMPLETE.md` - Answer generation
- `STEP8_COMPLETE.md` - LLM-as-Judge evaluation
- `STEP9_COMPLETE.md` - End-to-end pipeline
- `FILE_MANAGEMENT.md` - File handling and backups
- `GOLD_ANSWERS_EXPLAINED.md` - Gold labels concept

## ⚠️ Important Notes

### File Overwriting
Files use **deterministic naming** (no timestamps):
- Same configuration → Same filename → **Overwrites previous results**
- This is by design for clean organization
- Use git or backups to preserve important results

### API Keys
Never commit `.env` file:
```bash
# .gitignore already configured
.env
```

### Context Window
Ensure `K × chunk_size` fits LLM context:
```
k=5, chunk=512 → 2,560 tokens (safe for most models)
```

## 🤝 Contributing

This is a learning project. Suggestions welcome:
1. Test with your own documents
2. Share optimization findings
3. Report issues or improvements

## 📄 License

MIT License - Feel free to use and modify

## 🙏 Acknowledgments

Built as a systematic learning exercise for:
- RAG pipeline design
- Systematic evaluation methodology
- Production-ready ML engineering

---

**Ready to start?**

```bash
python src/run_pipeline.py --pdf data/raw/your_document.pdf --quick
```
