"""
Generate answers using retrieved context and LLM.

This module implements the generation step of the RAG pipeline:
1. Load QA pairs
2. Retrieve top-K chunks from vector index
3. Format context with chunk IDs
4. Generate grounded answers with citations using LLM
5. Save results for evaluation
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
import yaml

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
class GeneratedAnswer:
    """A generated answer with metadata."""
    question_id: int
    question: str
    question_type: str
    gold_answer: str
    gold_chunk_ids: List[int]
    retrieved_chunk_ids: List[int]
    retrieved_scores: List[float]
    context_chunks: List[Dict[str, Any]]  # List of {chunk_id, text}
    generated_answer: str
    cited_chunk_ids: List[int]  # Chunk IDs mentioned in answer
    chunk_config: str
    embedding_model: str
    llm_model: str
    k: int


class VectorIndex:
    """Load and query a FAISS vector index."""
    
    def __init__(self, index_dir: Path):
        """Load a FAISS index with its chunks and metadata."""
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
        
        self.chunk_config = index_dir.name.rsplit('__emb_', 1)[0]
        
        # Extract embedding model
        emb_meta = self.metadata['embedding_model']
        if isinstance(emb_meta, dict):
            self.embedding_model = emb_meta['model']
            self.embedding_id = emb_meta['id']
        else:
            self.embedding_model = emb_meta
            self.embedding_id = emb_meta
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search the index for top-K nearest neighbors.
        
        Returns:
            Tuple of (chunk_ids, scores, chunk_dicts)
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k)
        
        chunk_ids = []
        chunk_dicts = []
        
        for idx in indices[0]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                # Extract numeric ID
                chunk_id_str = chunk['id']
                parts = chunk_id_str.split('chunk')
                if len(parts) == 2:
                    chunk_id = int(parts[1])
                else:
                    chunk_id = idx
                
                chunk_ids.append(chunk_id)
                chunk_dicts.append({
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'token_count': chunk.get('token_count', 0)
                })
        
        return chunk_ids, scores[0].tolist(), chunk_dicts


class EmbeddingModel:
    """Embed queries using OpenAI API."""
    
    def __init__(self, model_name: str):
        """Initialize the embedding model."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text and return normalized vector."""
        if not isinstance(text, str):
            text = str(text)
        
        if len(text) > 8000:
            text = text[:8000]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class AnswerGenerator:
    """Generate answers using LLM with retrieved context."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """Initialize the answer generator."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        
        # Load prompt from file
        prompt_path = Path(__file__).parent.parent / "prompts" / "retriever_aware_answering.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read().strip()
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for the prompt."""
        context_lines = []
        for chunk in chunks:
            context_lines.append(f"[{chunk['chunk_id']}]: {chunk['text']}")
        return "\n\n".join(context_lines)
    
    def generate(self, question: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """
        Generate an answer with citations.
        
        Returns:
            Tuple of (answer_text, cited_chunk_ids)
        """
        context = self.format_context(context_chunks)
        prompt = self.prompt_template.format(question=question, context=context)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. You must respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=500
        )
        
        answer_text = response.choices[0].message.content
        if answer_text is None:
            answer_text = '{"answer": "No answer generated", "citations": []}'
        
        # Parse JSON response
        try:
            result = json.loads(answer_text.strip())
            answer = result.get("answer", "No answer generated")
            cited_ids = result.get("citations", [])
            
            # Validate citations are integers and in available chunks
            available_ids = [c['chunk_id'] for c in context_chunks]
            cited_ids = [int(cid) for cid in cited_ids if int(cid) in available_ids]
            
        except json.JSONDecodeError as e:
            console.print(f"[yellow]Warning: Failed to parse JSON response: {e}[/yellow]")
            console.print(f"[yellow]Raw response: {answer_text[:200]}...[/yellow]")
            # Fallback to old extraction method
            answer = answer_text
            cited_ids = self._extract_citations(answer_text, [c['chunk_id'] for c in context_chunks])
        
        return answer.strip(), cited_ids
    
    def _extract_citations(self, answer: str, available_ids: List[int]) -> List[int]:
        """Extract chunk IDs mentioned in the answer."""
        cited = []
        for chunk_id in available_ids:
            # Look for [chunk_id] or chunk_id patterns
            if f"[{chunk_id}]" in answer or f"chunk {chunk_id}" in answer.lower() or f"chunk_{chunk_id}" in answer:
                cited.append(chunk_id)
        return cited


def generate_answers_for_qa(
    index_dir: Path,
    qa_file: Path,
    output_file: Path,
    llm_model: str = "gpt-4o-mini",
    k: int = 5,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Generate answers for all QA pairs using a specific index.
    
    Args:
        index_dir: Directory containing the FAISS index
        qa_file: QA file with questions
        output_file: Output file for generated answers
        llm_model: OpenAI model for generation
        k: Number of chunks to retrieve
        temperature: LLM temperature
    
    Returns:
        Dictionary with generation statistics
    """
    # Load components
    console.print(f"[cyan]Loading index: {index_dir.name}[/cyan]")
    vector_index = VectorIndex(index_dir)
    embedder = EmbeddingModel(vector_index.embedding_model)
    generator = AnswerGenerator(model_name=llm_model, temperature=temperature)
    
    # Load QA pairs
    qa_items = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qa_items.append(QAItem(**data))
    
    console.print(f"[cyan]Processing {len(qa_items)} questions...[/cyan]")
    
    results = []
    stats = {
        "total_questions": len(qa_items),
        "successful": 0,
        "failed": 0,
        "avg_retrieved_chunks": 0.0,
        "avg_cited_chunks": 0.0
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(f"[cyan]Generating answers...", total=len(qa_items))
        
        for i, qa_item in enumerate(qa_items):
            try:
                # Embed question
                query_embedding = embedder.embed(qa_item.question)
                
                # Retrieve chunks
                retrieved_ids, scores, context_chunks = vector_index.search(query_embedding, k=k)
                
                # Generate answer
                answer, cited_ids = generator.generate(qa_item.question, context_chunks)
                
                # Store result
                result = GeneratedAnswer(
                    question_id=i,
                    question=qa_item.question,
                    question_type=qa_item.question_type,
                    gold_answer=qa_item.answer,
                    gold_chunk_ids=qa_item.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_scores=scores,
                    context_chunks=context_chunks,
                    generated_answer=answer,
                    cited_chunk_ids=cited_ids,
                    chunk_config=vector_index.chunk_config,
                    embedding_model=vector_index.embedding_id,
                    llm_model=llm_model,
                    k=k
                )
                
                results.append(result)
                stats["successful"] += 1
                stats["avg_retrieved_chunks"] += len(retrieved_ids)
                stats["avg_cited_chunks"] += len(cited_ids)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Failed on question {i}: {e}[/yellow]")
                stats["failed"] += 1
            
            progress.advance(task)
    
    # Calculate averages
    if stats["successful"] > 0:
        stats["avg_retrieved_chunks"] /= stats["successful"]
        stats["avg_cited_chunks"] /= stats["successful"]
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
    
    console.print(f"[green]✓ Saved {len(results)} answers to: {output_file}[/green]")
    
    return stats


def generate_answers_for_best_config(
    indexes_dir: Path,
    qa_dir: Path,
    output_dir: Path,
    best_config: str = "fy10syb__parser_pdfplumber__cs512__ov128",
    best_embedding: str = "ada-002",
    llm_model: str = "gpt-4o-mini",
    k: int = 5
) -> Dict[str, Any]:
    """
    Generate answers using the best performing configuration.
    
    Args:
        indexes_dir: Directory containing all FAISS indexes
        qa_dir: Directory containing QA files
        output_dir: Output directory for generated answers
        best_config: Best chunk configuration from evaluation
        best_embedding: Best embedding model from evaluation
        llm_model: LLM model for generation
        k: Number of chunks to retrieve
    
    Returns:
        Dictionary with generation statistics
    """
    # Find the best index directory
    index_dir = indexes_dir / f"{best_config}__emb_{best_embedding}"
    
    if not index_dir.exists():
        console.print(f"[red]Error: Index directory not found: {index_dir}[/red]")
        return {"error": "Index not found"}
    
    # Find corresponding QA file
    qa_file = qa_dir / f"{best_config}__qa.jsonl"
    
    if not qa_file.exists():
        console.print(f"[red]Error: QA file not found: {qa_file}[/red]")
        return {"error": "QA file not found"}
    
    console.print(Panel.fit(
        f"[bold cyan]Generating Answers[/bold cyan]\n"
        f"Config: [yellow]{best_config}[/yellow]\n"
        f"Embedding: [yellow]{best_embedding}[/yellow]\n"
        f"LLM: [yellow]{llm_model}[/yellow]\n"
        f"Retrieve: [yellow]Top-{k}[/yellow]",
        title="RAG Answer Generation",
        border_style="cyan"
    ))
    
    # Generate output filename
    output_file = output_dir / f"answers__{best_config}__emb_{best_embedding}__llm_{llm_model.replace('/', '_')}.jsonl"
    
    # Generate answers
    stats = generate_answers_for_qa(
        index_dir=index_dir,
        qa_file=qa_file,
        output_file=output_file,
        llm_model=llm_model,
        k=k
    )
    
    # Display summary
    console.print("\n[bold green]📊 Generation Summary:[/bold green]")
    console.print(f"  Total Questions: {stats['total_questions']}")
    console.print(f"  Successful: [green]{stats['successful']}[/green]")
    console.print(f"  Failed: [red]{stats['failed']}[/red]")
    console.print(f"  Avg Retrieved Chunks: {stats['avg_retrieved_chunks']:.1f}")
    console.print(f"  Avg Cited Chunks: {stats['avg_cited_chunks']:.1f}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate answers using RAG")
    parser.add_argument("--indexes-dir", type=Path, default=Path("indexes/faiss"),
                       help="Directory containing FAISS indexes (default: indexes/faiss)")
    parser.add_argument("--qa-dir", type=Path, default=Path("data/qa"),
                       help="Directory containing QA files (default: data/qa)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/generation"),
                       help="Output directory for answers (default: runs/generation)")
    parser.add_argument("--config", type=str, default="fy10syb__parser_pdfplumber__cs512__ov128",
                       help="Chunk configuration to use (default: best from eval)")
    parser.add_argument("--embedding", type=str, default="ada-002",
                       help="Embedding model to use (default: ada-002)")
    parser.add_argument("--llm", type=str, default="gpt-4o-mini",
                       help="LLM model for generation (default: gpt-4o-mini)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of chunks to retrieve (default: 5)")
    
    args = parser.parse_args()
    
    try:
        stats = generate_answers_for_best_config(
            indexes_dir=args.indexes_dir,
            qa_dir=args.qa_dir,
            output_dir=args.output_dir,
            best_config=args.config,
            best_embedding=args.embedding,
            llm_model=args.llm,
            k=args.k
        )
        
        if "error" not in stats:
            console.print(f"\n[bold green]✓ Answer generation complete![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
