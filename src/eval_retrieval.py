"""
Evaluate retrieval performance across multiple vector indexes.

This module loads synthetic QA pairs, queries vector indexes, and computes
retrieval metrics (Recall@K, Precision@K, MRR@K) to compare different
chunking and embedding configurations.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import faiss
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskID
from rich.table import Table
from rich.panel import Panel
import pandas as pd

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loading
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

console = Console()


@dataclass
class QAItem:
    """A synthetic question-answer pair with metadata."""
    question: str
    answer: str
    relevant_chunk_ids: List[int]  # Gold labels
    question_type: str
    difficulty: str
    chunk_config: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    question_id: int
    question: str
    question_type: str
    gold_chunk_ids: List[int]
    retrieved_chunk_ids: List[int]  # Top-K retrieved
    retrieved_scores: List[float]   # Similarity scores
    chunk_config: str
    embedding_model: str
    k: int


@dataclass
class MetricScores:
    """Computed metrics for a single query."""
    recall: float
    precision: float
    mrr: float  # Mean Reciprocal Rank


class VectorIndex:
    """Load and query a FAISS vector index."""
    
    def __init__(self, index_dir: Path):
        """
        Load a FAISS index with its chunks and metadata.
        
        Args:
            index_dir: Directory containing index.faiss, chunks.jsonl, metadata.json
        """
        self.index_dir = index_dir
        
        # Load FAISS index
        index_path = index_dir / "index.faiss"
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = index_dir / "chunks.jsonl"
        self.chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        # Load metadata
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.chunk_config = index_dir.name.rsplit('__emb_', 1)[0]  # Extract config
        
        # Extract embedding model name - could be string or dict
        emb_meta = self.metadata['embedding_model']
        if isinstance(emb_meta, dict):
            self.embedding_model = emb_meta['model']  # Use actual model name
            self.embedding_id = emb_meta['id']  # Keep ID for display
        else:
            self.embedding_model = emb_meta
            self.embedding_id = emb_meta
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Search the index for top-K nearest neighbors.
        
        Args:
            query_embedding: Query vector (normalized)
            k: Number of results to return
        
        Returns:
            Tuple of (chunk_ids, scores)
        """
        # Ensure query is 2D with shape (1, dim)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Extract chunk IDs from chunk metadata
        chunk_ids = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Extract numeric ID from string ID like "fy10syb__parserpymupdf__chunk0000"
                chunk_id_str = chunk['id']
                parts = chunk_id_str.split('chunk')
                if len(parts) == 2:
                    chunk_ids.append(int(parts[1]))
                else:
                    chunk_ids.append(idx)
            else:
                chunk_ids.append(idx)
        
        return chunk_ids, scores[0].tolist()


class EmbeddingModel:
    """Embed queries using OpenAI API."""
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: OpenAI embedding model name
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        # Test the model immediately to catch errors early
        try:
            test_response = self.client.embeddings.create(
                model=self.model_name,
                input=["test"]
            )
            console.print(f"[dim]✓ Embedding model {model_name} initialized ({len(test_response.data[0].embedding)}d)[/dim]")
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize {model_name}: {e}[/red]")
            raise
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Normalized embedding vector
        """
        # Ensure text is a string (not bytes or other type)
        if not isinstance(text, str):
            text = str(text)
        
        # Truncate very long texts to avoid API limits
        if len(text) > 8000:
            text = text[:8000]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]  # Always pass as list for consistency
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


def compute_metrics(retrieved_ids: List[int], gold_ids: List[int], k: int) -> MetricScores:
    """
    Compute retrieval metrics for a single query.
    
    Args:
        retrieved_ids: List of retrieved chunk IDs (top-K)
        gold_ids: List of gold standard chunk IDs
        k: Cutoff value for metrics
    
    Returns:
        MetricScores with recall, precision, and MRR
    """
    gold = set(gold_ids)
    topk = retrieved_ids[:k]
    
    # Intersection of retrieved and gold
    inter = [i for i in topk if i in gold]
    
    # Precision@K
    precision = len(inter) / k if k > 0 else 0.0
    
    # Recall@K
    recall = len(inter) / len(gold) if gold else 0.0
    
    # MRR@K (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, chunk_id in enumerate(topk, start=1):
        if chunk_id in gold:
            mrr = 1.0 / rank
            break
    
    return MetricScores(recall=recall, precision=precision, mrr=mrr)


def evaluate_index(
    index_dir: Path,
    qa_file: Path,
    k_values: List[int] = [1, 3, 5, 10]
) -> List[Dict[str, Any]]:
    """
    Evaluate a single vector index against QA pairs.
    
    Args:
        index_dir: Directory containing the FAISS index
        qa_file: QA file to evaluate against
        k_values: List of K values for metrics
    
    Returns:
        List of result dictionaries with metrics for each question
    """
    # Load index
    vector_index = VectorIndex(index_dir)
    
    # Initialize embedding model - use the actual model name string, not the dict
    embedder = EmbeddingModel(vector_index.embedding_model)
    
    # Load QA pairs
    qa_items = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qa_items.append(QAItem(**data))
    
    results = []
    
    for i, qa_item in enumerate(qa_items):
        # Embed the question
        query_embedding = embedder.embed(qa_item.question)
        
        # Search the index
        max_k = max(k_values)
        retrieved_ids, scores = vector_index.search(query_embedding, k=max_k)
        
        # Compute metrics at each K
        metrics_at_k = {}
        for k in k_values:
            metrics = compute_metrics(retrieved_ids, qa_item.relevant_chunk_ids, k)
            metrics_at_k[f"recall@{k}"] = metrics.recall
            metrics_at_k[f"precision@{k}"] = metrics.precision
            metrics_at_k[f"mrr@{k}"] = metrics.mrr
        
        # Store result
        result = {
            "question_id": i,
            "question": qa_item.question,
            "question_type": qa_item.question_type,
            "difficulty": qa_item.difficulty,
            "gold_chunk_ids": qa_item.relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_ids[:max_k],
            "scores": scores[:max_k],
            "chunk_config": vector_index.chunk_config,
            "embedding_model": vector_index.embedding_id,  # Use ID for display
            **metrics_at_k
        }
        results.append(result)
    
    return results


def evaluate_all_indexes(
    indexes_dir: Path,
    qa_dir: Path,
    output_dir: Path,
    k_values: List[int] = [1, 3, 5, 10]
) -> pd.DataFrame:
    """
    Evaluate all vector indexes against their corresponding QA files.
    
    Args:
        indexes_dir: Directory containing all FAISS indexes
        qa_dir: Directory containing QA files
        output_dir: Directory to save evaluation results
        k_values: List of K values for metrics
    
    Returns:
        DataFrame with aggregated results
    """
    # Find all index directories
    index_dirs = sorted([d for d in indexes_dir.iterdir() if d.is_dir() and (d / "index.faiss").exists()])
    
    if not index_dirs:
        console.print(f"[red]No index directories found in {indexes_dir}[/red]")
        return pd.DataFrame()
    
    console.print(Panel.fit(
        f"[bold cyan]Evaluating {len(index_dirs)} vector indexes[/bold cyan]\n"
        f"K values: {k_values}\n"
        f"Output: [yellow]{output_dir}[/yellow]",
        title="Retrieval Evaluation",
        border_style="cyan"
    ))
    
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Evaluating indexes...", total=len(index_dirs))
        
        for i, index_dir in enumerate(index_dirs, 1):
            # Extract chunk config from index directory name
            # e.g., "fy10syb__parser_pymupdf__cs256__ov64__emb_openai-small"
            dir_name = index_dir.name
            parts = dir_name.split("__emb_")
            if len(parts) != 2:
                console.print(f"[yellow]Skipping {dir_name} (unexpected format)[/yellow]")
                progress.advance(task)
                continue
            
            chunk_config = parts[0]
            
            # Find corresponding QA file
            qa_file = qa_dir / f"{chunk_config}__qa.jsonl"
            
            if not qa_file.exists():
                console.print(f"[yellow]QA file not found for {chunk_config}[/yellow]")
                progress.advance(task)
                continue
            
            progress.update(task, description=f"[cyan][{i}/{len(index_dirs)}] {dir_name[:50]}...")
            
            try:
                # Evaluate this index
                results = evaluate_index(index_dir, qa_file, k_values)
                all_results.extend(results)
                
            except Exception as e:
                console.print(f"[red]Error evaluating {dir_name}: {e}[/red]")
            
            progress.advance(task)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "retrieval_evaluation_detailed.csv"
    df.to_csv(output_file, index=False)
    console.print(f"\n[green]✓ Detailed results saved to:[/green] [cyan]{output_file}[/cyan]")
    
    # Compute summary statistics
    _display_summary(df, k_values)
    
    # Save summary by configuration
    summary_df = _compute_summary(df, k_values)
    summary_file = output_dir / "retrieval_evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    console.print(f"[green]✓ Summary saved to:[/green] [cyan]{summary_file}[/cyan]")
    
    return df


def _compute_summary(df: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    """Compute summary statistics by configuration."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by chunk_config and embedding_model
    summary_rows = []
    
    for (chunk_config, embedding_model), group in df.groupby(['chunk_config', 'embedding_model']):
        row = {
            'chunk_config': chunk_config,
            'embedding_model': embedding_model,
            'num_questions': len(group)
        }
        
        # Add mean metrics for each K
        for k in k_values:
            row[f'recall@{k}'] = group[f'recall@{k}'].mean()
            row[f'precision@{k}'] = group[f'precision@{k}'].mean()
            row[f'mrr@{k}'] = group[f'mrr@{k}'].mean()
        
        # Add by question type
        for qtype in group['question_type'].unique():
            qtype_group = group[group['question_type'] == qtype]
            row[f'{qtype}_count'] = len(qtype_group)
            for k in k_values:
                row[f'{qtype}_recall@{k}'] = qtype_group[f'recall@{k}'].mean()
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def _display_summary(df: pd.DataFrame, k_values: List[int]):
    """Display a summary table of evaluation results."""
    if df.empty:
        return
    
    console.print("\n")
    
    # Overall metrics table
    table = Table(title="📊 Overall Retrieval Performance", show_header=True, header_style="bold magenta")
    table.add_column("Configuration", style="cyan", no_wrap=False, width=40)
    table.add_column("Embedding", style="yellow", width=15)
    
    for k in k_values:
        table.add_column(f"R@{k}", justify="right", style="green")
        table.add_column(f"MRR@{k}", justify="right", style="blue")
    
    # Group by config and embedding
    for (chunk_config, embedding_model), group in df.groupby(['chunk_config', 'embedding_model']):
        row_data = [chunk_config, embedding_model]
        
        for k in k_values:
            recall = group[f'recall@{k}'].mean()
            mrr = group[f'mrr@{k}'].mean()
            row_data.append(f"{recall:.3f}")
            row_data.append(f"{mrr:.3f}")
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Best configurations
    console.print("\n[bold green]🏆 Top Configurations:[/bold green]")
    
    for k in [5]:  # Focus on K=5
        grouped = df.groupby(['chunk_config', 'embedding_model'])[f'recall@{k}'].mean()
        best_recall_idx = grouped.idxmax()
        best_mrr_idx = df.groupby(['chunk_config', 'embedding_model'])[f'mrr@{k}'].mean().idxmax()
        
        recall_score = grouped.max()
        mrr_score = df.groupby(['chunk_config', 'embedding_model'])[f'mrr@{k}'].mean().max()
        
        if isinstance(best_recall_idx, tuple):
            console.print(f"  Best Recall@{k}: [cyan]{best_recall_idx[0]}[/cyan] + [yellow]{best_recall_idx[1]}[/yellow] = [green]{recall_score:.3f}[/green]")
        if isinstance(best_mrr_idx, tuple):
            console.print(f"  Best MRR@{k}: [cyan]{best_mrr_idx[0]}[/cyan] + [yellow]{best_mrr_idx[1]}[/yellow] = [blue]{mrr_score:.3f}[/blue]")
    
    # By question type
    console.print("\n[bold magenta]📋 Performance by Question Type:[/bold magenta]")
    
    for qtype in df['question_type'].unique():
        qtype_df = df[df['question_type'] == qtype]
        grouped_qtype = qtype_df.groupby(['chunk_config', 'embedding_model'])['recall@5'].mean()
        best_config_idx = grouped_qtype.idxmax()
        best_score = grouped_qtype.max()
        
        if isinstance(best_config_idx, tuple):
            config_display = best_config_idx[0][:30] if len(best_config_idx[0]) > 30 else best_config_idx[0]
            console.print(f"  {qtype.capitalize():15} → [cyan]{config_display}[/cyan] + [yellow]{best_config_idx[1]}[/yellow] = [green]{best_score:.3f}[/green]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--indexes-dir", type=Path, default=Path("indexes/faiss"),
                       help="Directory containing FAISS indexes (default: indexes/faiss)")
    parser.add_argument("--qa-dir", type=Path, default=Path("data/qa"),
                       help="Directory containing QA files (default: data/qa)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/retrieval"),
                       help="Output directory for results (default: runs/retrieval)")
    parser.add_argument("--k-values", type=int, nargs='+', default=[1, 3, 5, 10],
                       help="K values for metrics (default: 1 3 5 10)")
    
    args = parser.parse_args()
    
    try:
        df = evaluate_all_indexes(
            indexes_dir=args.indexes_dir,
            qa_dir=args.qa_dir,
            output_dir=args.output_dir,
            k_values=args.k_values
        )
        
        console.print(f"\n[bold green]✓ Evaluation complete![/bold green] Evaluated {len(df)} queries.")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
