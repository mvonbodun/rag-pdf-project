Goal
Build a reproducible pipeline to (1) chunk long PDFs/markdown, (2) index with multiple embedding models, (3) generate synthetic QA with gold labels, (4) evaluate retrieval (Recall/MRR/Precision) across chunking × embeddings × retrieval/rerank combos, (5) attach an LLM generator and evaluate generation quality (faithfulness/relevance). Pick the best configuration and explain why.
0) Repo Layout
rag-eval/
  ├─ data/
  │   ├─ raw/                     # original PDFs / markdown dataset
  │   ├─ processed/               # parsed+chunked artifacts per config
  │   └─ qa/                      # synthetic QA sets per config (jsonl)
  ├─ indexes/
  │   ├─ faiss/                   # FAISS index per config+embedding
  │   └─ lancedb/                 # (optional) LanceDB tables
  ├─ runs/
  │   ├─ retrieval/               # CSV/JSON metrics per run
  │   └─ generation/              # LLM-as-judge results per run
  ├─ src/
  │   ├─ config.py                # Pydantic models for config/QA schema
  │   ├─ parse_chunk.py           # layout-aware parsing + chunking
  │   ├─ build_index.py           # embeddings + FAISS/LanceDB builders
  │   ├─ gen_synth_qa.py          # synthetic QA generation (LLM or rules)
  │   ├─ eval_retrieval.py        # Recall@K, Precision@K, MRR@K
  │   ├─ rerank.py                # Cohere / open-source reranking
  │   ├─ generate_answers.py      # attach generator LLM + prompts
  │   ├─ eval_generation.py       # faithfulness/grounding via LLM-judge
  │   ├─ utils_logging.py         # Logfire/Braintrust hooks
  │   └─ cli.py                   # orchestration / grid-search runner
  ├─ configs/
  │   ├─ grid.default.yaml        # chunk sizes/overlaps, embeddings, K
  │   └─ providers.yaml           # LiteLLM/OpenRouter/OpenAI keys+models
  ├─ requirements.txt
  └─ README.md  ← you are here
1) Dataset
Use either:
Kaggle “Enterprise RAG Markdown” (any.pdf → markdown)
or
Your own PDF(s) parsed into markdown.
Put source files under data/raw/.
2) Parsing & Chunking (Layout-aware)
Why: PDFs are visually structured; chunk along headings/paragraphs/tables first, then tokenize.
Implement (src/parse_chunk.py)
Parse to structured blocks using PyMuPDF or unstructured (headings, paragraphs, lists, tables).
Build chunks by unit of meaning, then token-constrain to your chunk size with overlap.
Export artifacts per configuration to data/processed/{docname}__cs{chunk}__ov{overlap}.jsonl.
Grid you’ll test (edit in configs/grid.default.yaml):
chunk_sizes: [128, 256, 512]
overlaps:    [32, 64, 128]
k_values:    [3, 5]            # top-K to evaluate
retrievers:  ["faiss"]         # ("lancedb" optional)
embeddings:
  - provider: "openai"         # via LiteLLM
    name: "text-embedding-3-small"
  - provider: "sentence-transformers"
    name: "all-MiniLM-L6-v2"
  - provider: "sentence-transformers"
    name: "instructor-large"   # or "hkunlp/instructor-large"
rerankers:
  - none
  - cohere
Constraint sanity check
Ensure k * chunk_size fits your generator’s context window (e.g., k=5, chunk=512 → 2560 tokens just for context chunks).
3) Indexing
Implement (src/build_index.py)
Batch-embed with ThreadPoolExecutor (fast, consistent).
Use LiteLLM for OpenAI/OpenRouter OR SentenceTransformers locally.
Build one index per (chunk_size, overlap, embedding_model):
Save FAISS under indexes/faiss/{doc}__cs{cs}__ov{ov}__emb{model_id}/.
(Optional) LanceDB under indexes/lancedb/....
4) Synthetic QA Generation (with Gold Labels)
Principle: Questions should intentionally probe overlaps and adjacent chunks to stress retrieval.
Implement (src/gen_synth_qa.py)
For every chunk, create a set of questions:
Factual (“What is X?”)
Analytical (“Why does X happen?”)
Comparative (“How does X differ from Y?”)
Summarization (“Summarize section Y”)
Multi-hop across 2–3 semantically related chunks.
Build gold labels as list of relevant chunk_ids (≥1).
For multi-chunk Qs, include all source chunk_ids.
For overlap stress, generate Qs whose answer spans the boundary between chunk N and N+1.
Store QA in data/qa/{doc}__cs{cs}__ov{ov}__emb{model_id}.jsonl.
Pydantic Schemas (src/config.py)
Validate QAItem:
class QAItem(BaseModel):
    id: str
    question: str
    relevant_chunk_ids: List[str]   # gold labels
    granularity: Literal["paragraph","section","page","multi-hop"]
    difficulty: Literal["easy","medium","hard"]
Enforce JSON correctness so your CLI vs JSON don’t drift.
Optional: Use an LLM to generate question variants; log prompts/outputs with Logfire (Pydantic Logfire) and/or Braintrust.
5) Retrieval Evaluation
Implement (src/eval_retrieval.py)
For each QA item:
Retrieve top-K chunks for the question string.
Compute:
Recall@K = fraction of gold chunk_ids present in top-K.
Precision@K = (# gold in top-K) / K.
MRR@K = 1 / rank(first gold in top-K), else 0.
Run for each config (chunk_size × overlap × embedding × retriever).
Save results to runs/retrieval/{stamp}__cs{cs}__ov{ov}__emb{model}.csv.
Optional Reranking (src/rerank.py)
Apply Cohere Reranker (or open-source) to top-M retrieved (e.g., M=20), then slice K.
Recompute metrics and compare with “no rerank”.
Report Template (you will fill numbers later)
Config	K	Recall@K	Precision@K	MRR@K
cs128-ov64 + MiniLM + FAISS	5			
cs256-ov64 + Instructor + FAISS	5			
...				
6) Select Best Retrieval Configuration(s)
Pick top 1–2 based on Recall@K and MRR@K (primary) plus Precision@K (secondary).
Note trade-offs: smaller chunks + high overlap → better recall, higher redundancy/cost; larger chunks → better semantic cohesion, but risk context blowout.
Document your choice and rationale.
7) Attach Generator + Prompting
Implement (src/generate_answers.py)
For each QA item:
Retrieve top-K chunks (using your selected config).
Construct a grounded prompt that:
Shows the question
Provides top-K chunks as citations with chunk_id markers
Instructs model to answer only from context; if not present, say “not found in context”
Save outputs to runs/generation/{stamp}__best_config.jsonl:
{
  "qa_id": "...",
  "question": "...",
  "retrieved_chunk_ids": ["..."],
  "answer": "...",
  "citations": ["chunk_12","chunk_13"]
}
Use LiteLLM to swap models (OpenAI/OpenRouter), and log every call via Logfire. (Optional: Braintrust for trace + thumbs up/down.)
8) Generation Evaluation (LLM-as-Judge)
Implement (src/eval_generation.py)
Judge criteria per answer:
Faithfulness (supported by retrieved text? cite spans/chunk_ids)
Relevance (addresses the question?)
Ground Truth Match (if gold textual answer is available; else skip)
Use a structured LLM-judge (judgy style) or your own prompt:
You are grading answers for QA over documents.
- Faithfulness: Is every claim supported by provided chunks? (0/1)
- Relevance: Does it answer the question? (0/1)
Return JSON: {"faithfulness": 0|1, "relevance": 0|1, "explanation": "..."}
Cross-check: if retrieval metrics high but faithfulness low, fix prompting.
Export runs/generation/{stamp}__judged.csv.
9) Orchestration (Grid Search)
Implement (src/cli.py)
End-to-end runner:
parse+chunk for each (cs, ov)
build index per embedding
generate QA for each config
eval retrieval (with/without rerank)
pick best
run generator + judge
Print a Rich table of metrics; write CSV/JSON in runs/.
CLI examples:
# 1) Parse+chunk all
python -m src.cli chunk --config configs/grid.default.yaml

# 2) Build indexes
python -m src.cli index --config configs/grid.default.yaml

# 3) Generate synthetic QA (LLM or rule-based)
python -m src.cli synth --config configs/grid.default.yaml

# 4) Evaluate retrieval (no rerank + rerank)
python -m src.cli eval-retrieval --config configs/grid.default.yaml --rerank none
python -m src.cli eval-retrieval --config configs/grid.default.yaml --rerank cohere

# 5) Select best & attach generator
python -m src.cli gen --best --k 5 --provider openai --model gpt-4o-mini

# 6) Judge answers
python -m src.cli eval-gen --judge openai:gpt-4o-mini
