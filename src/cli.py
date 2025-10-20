# End-to-end runner:
# 1. parse+chunk for each (cs, ov)
# 2. build index per embedding
# 3. generate QA for each config
# 4. eval retrieval (with/without rerank)
# 5. pick best
# 6. run generator + judge
# Print a Rich table of metrics; write CSV/JSON in runs/

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Evaluation CLI
- chunk:       Parse & chunk a single PDF with specified parameters
- chunk-grid:  Run complete grid of parsers × chunk_sizes × overlaps
- index:       Build vector index (BoW for dry-run, FAISS later)
- synth:       Generate synthetic QA with gold chunk_ids
- eval-retrieval: Compute Recall@K, Precision@K, MRR@K
- gen:         Produce grounded answers with citations
- eval-gen:    LLM-as-judge for faithfulness/relevance

The interface and artifacts (json/csv) remain compatible throughout the pipeline.
"""
import argparse, json, os, re, uuid, math, csv, time, yaml
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# Import our new parse_chunk module
from .parse_chunk import parse_and_chunk_pdf, PDFParser

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_QA = ROOT / "data" / "qa"
RUNS_RETR = ROOT / "runs" / "retrieval"
RUNS_GEN = ROOT / "runs" / "generation"
for p in [DATA_PROCESSED, DATA_QA, RUNS_RETR, RUNS_GEN]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
TOKEN_SPLIT = re.compile(r"\w+|[^\w\s]")
def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_SPLIT.findall(text)]

def chunks_with_overlap(tokens: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    if chunk_size <= 0: raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size: raise ValueError("0 <= overlap < chunk_size")
    out, i = [], 0
    step = chunk_size - overlap
    while i < len(tokens):
        out.append(tokens[i:i+chunk_size])
        i += step
    return out

def cosine_counter(a: Counter, b: Counter) -> float:
    # cosine similarity for sparse counters
    if not a or not b: return 0.0
    dot = sum(a[k]*b.get(k,0) for k in a)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return (dot/(na*nb)) if na and nb else 0.0

def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# ---------- 1) Chunk ----------
def cmd_chunk(args):
    """Parse and chunk a single PDF with specified parameters"""
    pdf_path = Path(args.doc)
    
    if not pdf_path.exists():
        raise SystemExit(f"Document not found: {pdf_path}")
    
    # Use the new parse_chunk module
    output_file = parse_and_chunk_pdf(
        pdf_path=pdf_path,
        parser=args.parser,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=DATA_PROCESSED,
        tokenizer=getattr(args, 'tokenizer', 'cl100k_base')
    )
    
    # Count chunks
    chunk_count = sum(1 for _ in output_file.open('r'))
    print(f"[chunk] ✓ Created {chunk_count} chunks → {output_file}")


def cmd_chunk_grid(args):
    """Run complete chunking grid from config file"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    
    # Load config
    with config_path.open('r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    docs = config['dataset']['docs']
    parsers = config['chunking']['parsers']
    tokenizer = config['chunking'].get('tokenizer', 'cl100k_base')
    
    # Handle both old format (chunk_sizes/overlaps) and new format (chunk_overlap_pairs)
    if 'chunk_overlap_pairs' in config['chunking']:
        # New paired format
        pairs = config['chunking']['chunk_overlap_pairs']
        chunk_sizes = [p['chunk_size'] for p in pairs]
        overlaps = [p['overlap'] for p in pairs]
        # Zip them together for pairing
        size_overlap_pairs = [(p['chunk_size'], p['overlap']) for p in pairs]
    else:
        # Old matrix format (all combinations)
        chunk_sizes = config['chunking']['chunk_sizes']
        overlaps = config['chunking']['overlaps']
        # Create all combinations
        size_overlap_pairs = [(cs, ov) for cs in chunk_sizes for ov in overlaps]
    
    # Calculate total combinations
    total_combinations = len(docs) * len(parsers) * len(size_overlap_pairs)
    
    if HAS_RICH and console:
        # Display grid info with Rich
        console.print("\n[bold cyan]═══ RAG Chunking Grid ═══[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Values", style="green")
        
        table.add_row("Documents", f"{len(docs)} files")
        table.add_row("Parsers", ", ".join(parsers))
        table.add_row("Chunk Size:Overlap Pairs", 
                     ", ".join([f"{cs}:{ov}" for cs, ov in size_overlap_pairs]))
        table.add_row("Tokenizer", tokenizer)
        table.add_row("[bold]Total Combinations[/bold]", f"[bold yellow]{total_combinations}[/bold yellow]")
        
        console.print(table)
        console.print()
    else:
        print(f"\n=== RAG Chunking Grid ===")
        print(f"Documents: {len(docs)} files")
        print(f"Parsers: {', '.join(parsers)}")
        print(f"Chunk Size:Overlap Pairs: {', '.join([f'{cs}:{ov}' for cs, ov in size_overlap_pairs])}")
        print(f"Total Combinations: {total_combinations}\n")
    
    # Track results
    results = []
    
    # Run grid with or without rich progress bar
    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Processing combinations...", total=total_combinations)
            results = _run_grid(docs, parsers, size_overlap_pairs, tokenizer, progress, task)
    else:
        print("Processing combinations...")
        results = _run_grid(docs, parsers, size_overlap_pairs, tokenizer, None, None)
    
    # Display results
    _display_results(results, total_combinations)


def _run_grid(docs, parsers, size_overlap_pairs, tokenizer, progress, task):
    """Internal function to run the grid"""
    results = []
    
    for doc_path_str in docs:
        doc_path = Path(doc_path_str)
        
        if not doc_path.exists():
            print(f"⚠ Warning: Document not found: {doc_path}")
            skip_count = len(parsers) * len(size_overlap_pairs)
            if progress:
                progress.update(task, advance=skip_count)
            continue
        
        for parser in parsers:
            for chunk_size, overlap in size_overlap_pairs:
                # Update progress description
                if progress:
                    desc = f"[cyan]{doc_path.name} | {parser} | cs={chunk_size} ov={overlap}"
                    progress.update(task, description=desc)
                else:
                    print(f"  Processing: {doc_path.name} | {parser} | cs={chunk_size} ov={overlap}")
                
                try:
                    # Run parsing and chunking
                    start_time = time.time()
                    output_file = parse_and_chunk_pdf(
                        pdf_path=doc_path,
                        parser=parser,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        output_dir=DATA_PROCESSED,
                        tokenizer=tokenizer
                    )
                    elapsed = time.time() - start_time
                    
                    # Count chunks
                    chunk_count = sum(1 for _ in output_file.open('r'))
                    
                    results.append({
                        'doc': doc_path.name,
                        'parser': parser,
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'chunks': chunk_count,
                        'time': elapsed,
                        'output': output_file.name
                    })
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    results.append({
                        'doc': doc_path.name,
                        'parser': parser,
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'chunks': 0,
                        'time': 0,
                        'error': str(e)
                    })
                
                if progress:
                    progress.update(task, advance=1)
    
    return results


def _display_results(results, total_combinations):
    """Display results summary"""
    if HAS_RICH and console:
        console.print("\n[bold green]═══ Chunking Results ═══[/bold green]\n")
        
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Parser", style="cyan")
        results_table.add_column("Chunk Size", justify="right", style="yellow")
        results_table.add_column("Overlap", justify="right", style="yellow")
        results_table.add_column("Chunks", justify="right", style="green")
        results_table.add_column("Time (s)", justify="right", style="blue")
        results_table.add_column("Output File", style="dim")
        
        for result in results:
            if 'error' in result:
                results_table.add_row(
                    result['parser'],
                    str(result['chunk_size']),
                    str(result['overlap']),
                    "[red]ERROR[/red]",
                    "-",
                    result.get('error', 'Unknown error')
                )
            else:
                results_table.add_row(
                    result['parser'],
                    str(result['chunk_size']),
                    str(result['overlap']),
                    str(result['chunks']),
                    f"{result['time']:.2f}",
                    result['output']
                )
        
        console.print(results_table)
        
        # Summary statistics
        console.print("\n[bold cyan]═══ Summary Statistics ═══[/bold cyan]\n")
        
        total_chunks = sum(r['chunks'] for r in results if 'error' not in r)
        total_time = sum(r['time'] for r in results if 'error' not in r)
        successful = sum(1 for r in results if 'error' not in r)
        failed = sum(1 for r in results if 'error' in r)
        
        summary_table = Table(show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Combinations", str(total_combinations))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(failed))
        summary_table.add_row("Total Chunks Created", f"{total_chunks:,}")
        summary_table.add_row("Total Time", f"{total_time:.2f}s")
        summary_table.add_row("Average Time per Combo", f"{total_time/max(successful, 1):.2f}s")
        
        console.print(summary_table)
        
        # Parser comparison at standard config
        console.print("\n[bold cyan]═══ Parser Comparison (at cs=256, ov=64) ═══[/bold cyan]\n")
        
        comparison_table = Table(show_header=True, header_style="bold magenta")
        comparison_table.add_column("Parser", style="cyan")
        comparison_table.add_column("Chunks", justify="right", style="green")
        comparison_table.add_column("Difference", justify="right", style="yellow")
        
        baseline_chunks = None
        for result in results:
            if result['chunk_size'] == 256 and result['overlap'] == 64 and 'error' not in result:
                chunks = result['chunks']
                if result['parser'] == 'pymupdf':
                    baseline_chunks = chunks
                    comparison_table.add_row(result['parser'], str(chunks), "baseline")
                elif baseline_chunks:
                    diff = chunks - baseline_chunks
                    diff_pct = (diff / baseline_chunks) * 100
                    comparison_table.add_row(
                        result['parser'],
                        str(chunks),
                        f"{diff:+d} ({diff_pct:+.1f}%)"
                    )
        
        console.print(comparison_table)
        console.print("\n[bold green]✓ Chunking grid complete![/bold green]\n")
    else:
        # Plain text output
        print("\n=== Chunking Results ===\n")
        for result in results:
            if 'error' in result:
                print(f"  {result['parser']:12} cs={result['chunk_size']:3} ov={result['overlap']:3} ERROR: {result['error']}")
            else:
                print(f"  {result['parser']:12} cs={result['chunk_size']:3} ov={result['overlap']:3} chunks={result['chunks']:4} time={result['time']:.2f}s")
        
        total_chunks = sum(r['chunks'] for r in results if 'error' not in r)
        total_time = sum(r['time'] for r in results if 'error' not in r)
        print(f"\n✓ Total chunks created: {total_chunks:,} in {total_time:.2f}s\n")

# ---------- 2) Index (BoW counters as a stand-in for embeddings) ----------
def build_bow_index(chunks: List[Dict]) -> Tuple[List[str], List[Counter]]:
    ids, vecs = [], []
    for c in chunks:
        ids.append(c["id"])
        vecs.append(Counter(tokenize(c["text"])))
    return ids, vecs

def cmd_index(args):
    # Support new naming format with parser
    parser = getattr(args, 'parser', 'pymupdf')  # default to pymupdf
    proc = DATA_PROCESSED / f"{Path(args.doc).stem}__parser_{parser}__cs{args.chunk_size}__ov{args.overlap}.jsonl"
    
    # Fall back to old naming if new format not found
    if not proc.exists():
        proc = DATA_PROCESSED / f"{Path(args.doc).stem}__cs{args.chunk_size}__ov{args.overlap}.jsonl"
    
    if not proc.exists():
        raise SystemExit(f"Processed file not found: {proc}. Run 'chunk' first.")
    
    chunks = [json.loads(l) for l in proc.read_text(encoding="utf-8").splitlines()]
    ids, vecs = build_bow_index(chunks)

    out_dir = ROOT / "indexes" / "dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{proc.stem}__dry_bow.json"
    with out_fp.open("w", encoding="utf-8") as f:
        json.dump({"ids": ids, "chunks_file": str(proc), "bow": [dict(v) for v in vecs]}, f)

    print(f"[index] wrote BoW index → {out_fp}")

# ---------- 3) Synthetic QA (multi-type, incl. overlap boundary) ----------
def cmd_synth(args):
    proc = DATA_PROCESSED / f"{Path(args.doc).stem}__cs{args.chunk_size}__ov{args.overlap}.jsonl"
    if not proc.exists():
        raise SystemExit(f"Processed file not found: {proc}. Run 'chunk' first.")
    chunks = [json.loads(l) for l in proc.read_text(encoding="utf-8").splitlines()]

    qa_items = []
    # Basic strategy:
    # - factual: per chunk
    # - comparative/analytical: every 2 consecutive chunks (overlap stress)
    # - multi-hop: every 3rd chunk includes previous/next chunk ids
    for i, ch in enumerate(chunks):
        # Factual
        qa_items.append({
            "id": str(uuid.uuid4()),
            "question": f"What is the main idea of chunk {i}?",
            "relevant_chunk_ids": [ch["id"]],
            "granularity": "paragraph",
            "difficulty": "easy"
        })
        # Overlap stress (pair)
        if i+1 < len(chunks):
            qa_items.append({
                "id": str(uuid.uuid4()),
                "question": f"Explain the connection between chunk {i} and chunk {i+1}.",
                "relevant_chunk_ids": [ch["id"], chunks[i+1]["id"]],
                "granularity": "section",
                "difficulty": "medium"
            })
        # Multi-hop
        if i % 3 == 0 and i+2 < len(chunks):
            qa_items.append({
                "id": str(uuid.uuid4()),
                "question": f"Summarize the theme spanning chunks {i}-{i+2}.",
                "relevant_chunk_ids": [chunks[i]["id"], chunks[i+1]["id"], chunks[i+2]["id"]],
                "granularity": "multi-hop",
                "difficulty": "hard"
            })

    out_file = DATA_QA / f"{proc.stem}__qa.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for q in qa_items:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[synth] wrote {len(qa_items)} QA items → {out_file}")

# ---------- 4) Retrieval Eval ----------
def load_index(doc, cs, ov):
    idx_fp = ROOT / "indexes" / "dryrun" / f"{Path(doc).stem}__cs{cs}__ov{ov}__dry_bow.json"
    if not idx_fp.exists():
        raise SystemExit(f"Index not found: {idx_fp}. Run 'index' first.")
    blob = json.loads(idx_fp.read_text(encoding="utf-8"))
    ids = blob["ids"]
    vecs = [Counter(v) for v in blob["bow"]]
    proc_file = Path(blob["chunks_file"])
    chunks = [json.loads(l) for l in proc_file.read_text(encoding="utf-8").splitlines()]
    id2text = {c["id"]: c["text"] for c in chunks}
    return ids, vecs, id2text

def bow_search(query, ids, vecs, top_k):
    qv = Counter(tokenize(query))
    sims = [(ids[i], cosine_counter(qv, vecs[i])) for i in range(len(ids))]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sims[:top_k]]

def compute_metrics(topk_ids, gold_ids, K):
    gold = set(gold_ids)
    topk = topk_ids[:K]
    inter = [i for i in topk if i in gold]
    prec = len(inter)/K
    rec  = len(inter)/len(gold) if gold else 0.0
    mrr  = 0.0
    for r, cid in enumerate(topk, start=1):
        if cid in gold:
            mrr = 1.0/r
            break
    return prec, rec, mrr

def cmd_eval_retrieval(args):
    ids, vecs, _ = load_index(args.doc, args.chunk_size, args.overlap)
    qa_fp = DATA_QA / f"{Path(args.doc).stem}__cs{args.chunk_size}__ov{args.overlap}__qa.jsonl"
    if not qa_fp.exists():
        raise SystemExit(f"QA file not found: {qa_fp}. Run 'synth' first.")
    qa_items = [json.loads(l) for l in qa_fp.read_text(encoding="utf-8").splitlines()]

    os.makedirs(RUNS_RETR, exist_ok=True)
    stamp = now_stamp()
    out_csv = RUNS_RETR / f"retrieval__{Path(args.doc).stem}__cs{args.chunk_size}__ov{args.overlap}__K{args.k}__{stamp}.csv"

    rows = []
    for qa in qa_items:
        topk = bow_search(qa["question"], ids, vecs, top_k=args.k)
        prec, rec, mrr = compute_metrics(topk, qa["relevant_chunk_ids"], args.k)
        rows.append({
            "qa_id": qa["id"],
            "K": args.k,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "mrr": round(mrr, 4),
            "gold": "|".join(qa["relevant_chunk_ids"]),
            "topk": "|".join(topk)
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # print aggregate
    avg_p = sum(r["precision"] for r in rows)/len(rows)
    avg_r = sum(r["recall"] for r in rows)/len(rows)
    avg_m = sum(r["mrr"] for r in rows)/len(rows)
    print(f"[eval-retrieval] K={args.k} avg P={avg_p:.3f} R={avg_r:.3f} MRR={avg_m:.3f}")
    print(f"[eval-retrieval] wrote → {out_csv}")

# ---------- 5) Generate answers (simple, grounded) ----------
def cmd_gen(args):
    ids, vecs, id2text = load_index(args.doc, args.chunk_size, args.overlap)
    qa_fp = DATA_QA / f"{Path(args.doc).stem}__cs{args.chunk_size}__ov{args.overlap}__qa.jsonl"
    qa_items = [json.loads(l) for l in qa_fp.read_text(encoding="utf-8").splitlines()]
    stamp = now_stamp()
    out_jsonl = RUNS_GEN / f"answers__{Path(args.doc).stem}__cs{args.chunk_size}__ov{args.overlap}__K{args.k}__{stamp}.jsonl"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for qa in qa_items:
            topk = bow_search(qa["question"], ids, vecs, top_k=args.k)
            # naive extractive “answer”: first sentence from top chunk
            ctx = id2text[topk[0]] if topk else ""
            sent = re.split(r"(?<=[.!?])\s+", ctx.strip())
            answer = sent[0] if sent and sent[0] else (ctx[:200] if ctx else "Answer not found in context.")
            rec = {
                "qa_id": qa["id"],
                "question": qa["question"],
                "retrieved_chunk_ids": topk,
                "answer": answer,
                "citations": topk[:2]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[gen] wrote answers → {out_jsonl}")

# ---------- 6) Evaluate generation (rule-based judge) ----------
def cmd_eval_gen(args):
    # Faithfulness: answer’s tokens ⊆ union(topK texts) to a threshold
    # Relevance: token overlap with the question to a small threshold
    ans_path = Path(args.answers)
    if not ans_path.exists():
        raise SystemExit(f"Answers file not found: {ans_path}")

    # infer doc/cs/ov from filename to reload context
    m = re.search(r"answers__(.+)__cs(\d+)__ov(\d+)__K(\d+)", ans_path.stem)
    if not m:
        raise SystemExit("Cannot parse answers filename for doc/cs/ov/K.")
    doc_stem, cs, ov, K = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    doc_guess = DATA_RAW / f"{doc_stem}.md"
    if not doc_guess.exists():
        # try .pdf fallback name (still fine for dry run)
        doc_guess = DATA_RAW / f"{doc_stem}.pdf"

    ids, vecs, id2text = load_index(doc_guess, cs, ov)

    judged_rows = []
    items = [json.loads(l) for l in ans_path.read_text(encoding="utf-8").splitlines()]
    for it in items:
        answer = it["answer"]
        ctx_text = " ".join(id2text.get(cid, "") for cid in it["retrieved_chunk_ids"])
        a_toks = set(tokenize(answer))
        c_toks = set(tokenize(ctx_text))
        q_toks = set(tokenize(it["question"]))

        # simple thresholds
        faith = 1 if len(a_toks & c_toks) / max(1, len(a_toks)) >= 0.6 else 0
        rel = 1 if len(q_toks & a_toks) >= 2 else 0

        judged_rows.append({
            "qa_id": it["qa_id"],
            "faithfulness": faith,
            "relevance": rel,
            "explanation": "rule-based dry-run judge",
        })

    out_csv = RUNS_GEN / f"judged__{doc_stem}__cs{cs}__ov{ov}__K{K}__{now_stamp()}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["qa_id","faithfulness","relevance","explanation"])
        w.writeheader()
        w.writerows(judged_rows)

    avg_f = sum(r["faithfulness"] for r in judged_rows)/len(judged_rows)
    avg_r = sum(r["relevance"] for r in judged_rows)/len(judged_rows)
    print(f"[eval-gen] avg faithfulness={avg_f:.3f} relevance={avg_r:.3f}")
    print(f"[eval-gen] wrote → {out_csv}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="RAG Evaluation CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Chunk single file
    p_chunk = sub.add_parser("chunk", help="Parse and chunk a single PDF")
    p_chunk.add_argument("--doc", required=True, help="Path to PDF file")
    p_chunk.add_argument("--parser", choices=["pymupdf", "unstructured", "pdfplumber"], 
                        default="pymupdf", help="Which parser to use")
    p_chunk.add_argument("--chunk-size", type=int, default=256, help="Chunk size in tokens")
    p_chunk.add_argument("--overlap", type=int, default=64, help="Overlap in tokens")
    p_chunk.add_argument("--tokenizer", default="cl100k_base", help="Tokenizer encoding")
    p_chunk.set_defaults(func=cmd_chunk)

    # Chunk grid
    p_chunk_grid = sub.add_parser("chunk-grid", help="Run complete chunking grid from config")
    p_chunk_grid.add_argument("--config", default="configs/grid.default.yaml", 
                             help="Path to config YAML file")
    p_chunk_grid.set_defaults(func=cmd_chunk_grid)

    p_index = sub.add_parser("index", help="Build vector index")
    p_index.add_argument("--doc", required=True)
    p_index.add_argument("--parser", choices=["pymupdf", "unstructured", "pdfplumber"], 
                        default="pymupdf", help="Which parser was used for chunking")
    p_index.add_argument("--chunk-size", type=int, default=256)
    p_index.add_argument("--overlap", type=int, default=64)
    p_index.set_defaults(func=cmd_index)

    p_synth = sub.add_parser("synth", help="Generate synthetic QA")
    p_synth.add_argument("--doc", required=True)
    p_synth.add_argument("--chunk-size", type=int, default=256)
    p_synth.add_argument("--overlap", type=int, default=64)
    p_synth.set_defaults(func=cmd_synth)

    p_evalr = sub.add_parser("eval-retrieval", help="Evaluate retrieval metrics")
    p_evalr.add_argument("--doc", required=True)
    p_evalr.add_argument("--chunk-size", type=int, default=256)
    p_evalr.add_argument("--overlap", type=int, default=64)
    p_evalr.add_argument("--k", type=int, default=5)
    p_evalr.set_defaults(func=cmd_eval_retrieval)

    p_gen = sub.add_parser("gen", help="Generate answers")
    p_gen.add_argument("--doc", required=True)
    p_gen.add_argument("--chunk-size", type=int, default=256)
    p_gen.add_argument("--overlap", type=int, default=64)
    p_gen.add_argument("--k", type=int, default=5)
    p_gen.set_defaults(func=cmd_gen)

    p_evalg = sub.add_parser("eval-gen", help="Evaluate generation quality")
    p_evalg.add_argument("--answers", required=True, help="Path to answers__...jsonl produced by gen")
    p_evalg.set_defaults(func=cmd_eval_gen)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
