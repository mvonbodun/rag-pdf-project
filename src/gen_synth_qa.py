"""
Generate synthetic QA pairs from chunked documents.

This module creates diverse question types to evaluate retrieval systems:
- Factual: Single-chunk retrieval
- Analytical: Reasoning-based questions
- Multi-hop: Questions spanning multiple chunks
- Boundary: Questions testing chunk overlap handling
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

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
    relevant_chunk_ids: List[int]  # Chunk indices that contain the answer
    question_type: str  # factual, analytical, multi-hop, boundary
    difficulty: str  # easy, medium, hard
    chunk_config: str  # e.g., "pymupdf__cs256__ov64"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Chunk:
    """A text chunk from the chunked document."""
    id: str  # e.g., "fy10syb__parserpymupdf__chunk0000"
    text: str
    token_count: int
    source_blocks: List[int]
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def chunk_id(self) -> int:
        """Extract numeric chunk ID from the string ID."""
        # Extract the number from "chunk0000"
        parts = self.id.split('chunk')
        if len(parts) == 2:
            return int(parts[1])
        return 0


class QAGenerator:
    """Generate synthetic QA pairs using LLM."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the QA generator.
        
        Args:
            model_name: OpenAI model to use for generation
            temperature: Sampling temperature (0.0-1.0)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
    
    def load_chunks(self, chunk_file: Path) -> List[Chunk]:
        """Load chunks from a JSONL file."""
        chunks = []
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                chunks.append(Chunk(**data))
        return chunks
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the LLM with a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    
    def generate_factual_question(self, chunk: Chunk) -> Optional[QAItem]:
        """
        Generate a factual question answerable from a single chunk.
        
        These test basic retrieval: "What is X?" or "Who did Y?"
        """
        system_prompt = (
            "You are an expert at creating factual questions from text. "
            "Generate a clear, specific question that can be answered directly from the given text. "
            "The question should test comprehension and retrieval of key information."
        )
        
        prompt = f"""Based on the following text, generate ONE factual question and its answer.

TEXT:
{chunk.text}

Respond in this exact JSON format:
{{
  "question": "The factual question",
  "answer": "The answer extracted from the text",
  "difficulty": "easy|medium|hard"
}}"""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            return QAItem(
                question=data["question"],
                answer=data["answer"],
                relevant_chunk_ids=[chunk.chunk_id],
                question_type="factual",
                difficulty=data.get("difficulty", "medium"),
                chunk_config="",  # Will be set by caller
                metadata={"source_chunk_id": chunk.chunk_id}
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate factual question: {e}[/yellow]")
            return None
    
    def generate_analytical_question(self, chunk: Chunk) -> Optional[QAItem]:
        """
        Generate an analytical question requiring reasoning.
        
        These test understanding: "Why did X happen?" or "What are the implications of Y?"
        """
        system_prompt = (
            "You are an expert at creating analytical questions from text. "
            "Generate a question that requires reasoning, inference, or analysis beyond just "
            "extracting facts. The answer should require understanding the text's implications."
        )
        
        prompt = f"""Based on the following text, generate ONE analytical question and its answer.

TEXT:
{chunk.text}

Respond in this exact JSON format:
{{
  "question": "The analytical question",
  "answer": "The answer requiring reasoning",
  "difficulty": "easy|medium|hard"
}}"""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            return QAItem(
                question=data["question"],
                answer=data["answer"],
                relevant_chunk_ids=[chunk.chunk_id],
                question_type="analytical",
                difficulty=data.get("difficulty", "medium"),
                chunk_config="",
                metadata={"source_chunk_id": chunk.chunk_id}
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate analytical question: {e}[/yellow]")
            return None
    
    def generate_multi_hop_question(self, chunks: List[Chunk]) -> Optional[QAItem]:
        """
        Generate a multi-hop question requiring information from multiple chunks.
        
        These test complex retrieval: questions whose answers span multiple chunks.
        """
        if len(chunks) < 2:
            return None
        
        system_prompt = (
            "You are an expert at creating complex questions that require synthesizing "
            "information from multiple text passages. Generate a question whose answer "
            "requires connecting facts or ideas from ALL the given passages."
        )
        
        combined_text = "\n\n---\n\n".join([f"PASSAGE {i+1}:\n{c.text}" for i, c in enumerate(chunks)])
        
        prompt = f"""Based on the following text passages, generate ONE question that requires information from MULTIPLE passages to answer.

{combined_text}

Respond in this exact JSON format:
{{
  "question": "The multi-hop question",
  "answer": "The answer synthesizing multiple passages",
  "difficulty": "medium|hard"
}}"""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            return QAItem(
                question=data["question"],
                answer=data["answer"],
                relevant_chunk_ids=[c.chunk_id for c in chunks],
                question_type="multi-hop",
                difficulty=data.get("difficulty", "hard"),
                chunk_config="",
                metadata={"source_chunk_ids": [c.chunk_id for c in chunks]}
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate multi-hop question: {e}[/yellow]")
            return None
    
    def generate_boundary_question(self, chunk1: Chunk, chunk2: Chunk) -> Optional[QAItem]:
        """
        Generate a question testing chunk overlap handling.
        
        These test boundary cases: information split across adjacent chunks.
        """
        system_prompt = (
            "You are an expert at creating questions that test information boundaries. "
            "Generate a question whose answer spans the boundary between two adjacent text passages. "
            "The answer should require information from BOTH passages."
        )
        
        prompt = f"""Based on these TWO ADJACENT text passages, generate ONE question whose answer spans the boundary between them.

PASSAGE 1:
{chunk1.text}

PASSAGE 2:
{chunk2.text}

Respond in this exact JSON format:
{{
  "question": "The boundary question",
  "answer": "The answer spanning both passages",
  "difficulty": "medium|hard"
}}"""
        
        try:
            response = self._call_llm(prompt, system_prompt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response)
            return QAItem(
                question=data["question"],
                answer=data["answer"],
                relevant_chunk_ids=[chunk1.chunk_id, chunk2.chunk_id],
                question_type="boundary",
                difficulty=data.get("difficulty", "medium"),
                chunk_config="",
                metadata={"source_chunk_ids": [chunk1.chunk_id, chunk2.chunk_id]}
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to generate boundary question: {e}[/yellow]")
            return None


def generate_qa_for_chunks(
    chunk_file: Path,
    output_file: Path,
    model_name: str = "gpt-4o-mini",
    num_factual: int = 5,
    num_analytical: int = 3,
    num_multi_hop: int = 2,
    num_boundary: int = 2
) -> Dict[str, Any]:
    """
    Generate synthetic QA pairs for a chunked document.
    
    Args:
        chunk_file: Path to the chunked JSONL file
        output_file: Path to save QA pairs (JSONL format)
        model_name: OpenAI model to use
        num_factual: Number of factual questions to generate
        num_analytical: Number of analytical questions to generate
        num_multi_hop: Number of multi-hop questions to generate
        num_boundary: Number of boundary questions to generate
    
    Returns:
        Dictionary with generation statistics
    """
    generator = QAGenerator(model_name=model_name)
    chunks = generator.load_chunks(chunk_file)
    
    # Extract chunk config from filename: {doc}__parser_{parser}__cs{cs}__ov{ov}.jsonl
    chunk_config = chunk_file.stem  # Remove .jsonl extension
    
    qa_items = []
    stats = {
        "total_chunks": len(chunks),
        "generated": {"factual": 0, "analytical": 0, "multi-hop": 0, "boundary": 0},
        "failed": {"factual": 0, "analytical": 0, "multi-hop": 0, "boundary": 0}
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Generate factual questions
        task = progress.add_task(f"[cyan]Factual questions...", total=num_factual)
        for i in range(min(num_factual, len(chunks))):
            chunk = chunks[i]
            qa_item = generator.generate_factual_question(chunk)
            if qa_item:
                qa_item.chunk_config = chunk_config
                qa_items.append(qa_item)
                stats["generated"]["factual"] += 1
            else:
                stats["failed"]["factual"] += 1
            progress.advance(task)
        
        # Generate analytical questions
        task = progress.add_task(f"[magenta]Analytical questions...", total=num_analytical)
        for i in range(min(num_analytical, len(chunks))):
            # Use different chunks than factual
            chunk = chunks[len(chunks) // 2 + i] if len(chunks) > num_factual + i else chunks[i]
            qa_item = generator.generate_analytical_question(chunk)
            if qa_item:
                qa_item.chunk_config = chunk_config
                qa_items.append(qa_item)
                stats["generated"]["analytical"] += 1
            else:
                stats["failed"]["analytical"] += 1
            progress.advance(task)
        
        # Generate multi-hop questions
        task = progress.add_task(f"[green]Multi-hop questions...", total=num_multi_hop)
        for i in range(num_multi_hop):
            # Select 2-3 consecutive chunks
            if i * 3 + 2 < len(chunks):
                chunk_group = chunks[i * 3:i * 3 + 3]
            elif len(chunks) >= 2:
                chunk_group = chunks[-2:]
            else:
                progress.advance(task)
                stats["failed"]["multi-hop"] += 1
                continue
            
            qa_item = generator.generate_multi_hop_question(chunk_group)
            if qa_item:
                qa_item.chunk_config = chunk_config
                qa_items.append(qa_item)
                stats["generated"]["multi-hop"] += 1
            else:
                stats["failed"]["multi-hop"] += 1
            progress.advance(task)
        
        # Generate boundary questions
        task = progress.add_task(f"[yellow]Boundary questions...", total=num_boundary)
        for i in range(min(num_boundary, len(chunks) - 1)):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            qa_item = generator.generate_boundary_question(chunk1, chunk2)
            if qa_item:
                qa_item.chunk_config = chunk_config
                qa_items.append(qa_item)
                stats["generated"]["boundary"] += 1
            else:
                stats["failed"]["boundary"] += 1
            progress.advance(task)
    
    # Save QA items
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa_item in qa_items:
            f.write(json.dumps(asdict(qa_item), ensure_ascii=False) + '\n')
    
    stats["total_generated"] = sum(stats["generated"].values())
    stats["total_failed"] = sum(stats["failed"].values())
    stats["output_file"] = str(output_file)
    stats["chunk_config"] = chunk_config
    
    return stats


def generate_qa_grid(
    chunk_dir: Path,
    output_dir: Path,
    model_name: str = "gpt-4o-mini",
    num_factual: int = 5,
    num_analytical: int = 3,
    num_multi_hop: int = 2,
    num_boundary: int = 2
) -> List[Dict[str, Any]]:
    """
    Generate QA pairs for all chunked files in a directory.
    
    Args:
        chunk_dir: Directory containing chunked JSONL files
        output_dir: Directory to save QA files
        model_name: OpenAI model to use
        num_factual: Number of factual questions per file
        num_analytical: Number of analytical questions per file
        num_multi_hop: Number of multi-hop questions per file
        num_boundary: Number of boundary questions per file
    
    Returns:
        List of statistics for each file processed
    """
    chunk_files = sorted(chunk_dir.glob("*.jsonl"))
    
    if not chunk_files:
        console.print(f"[red]No JSONL files found in {chunk_dir}[/red]")
        return []
    
    console.print(Panel.fit(
        f"[bold cyan]Generating QA pairs for {len(chunk_files)} chunk files[/bold cyan]\n"
        f"Model: [yellow]{model_name}[/yellow]\n"
        f"Questions per file: {num_factual} factual, {num_analytical} analytical, "
        f"{num_multi_hop} multi-hop, {num_boundary} boundary",
        title="QA Generation Grid",
        border_style="cyan"
    ))
    
    all_stats = []
    
    for i, chunk_file in enumerate(chunk_files, 1):
        console.print(f"\n[bold][{i}/{len(chunk_files)}][/bold] Processing: [cyan]{chunk_file.name}[/cyan]")
        
        # Output file: data/qa/{same_name}__qa.jsonl
        output_file = output_dir / f"{chunk_file.stem}__qa.jsonl"
        
        try:
            stats = generate_qa_for_chunks(
                chunk_file=chunk_file,
                output_file=output_file,
                model_name=model_name,
                num_factual=num_factual,
                num_analytical=num_analytical,
                num_multi_hop=num_multi_hop,
                num_boundary=num_boundary
            )
            all_stats.append(stats)
            
            # Display stats
            console.print(f"  ✓ Generated [green]{stats['total_generated']}[/green] questions")
            console.print(f"  → Saved to: [dim]{output_file}[/dim]")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            all_stats.append({
                "chunk_config": chunk_file.stem,
                "error": str(e),
                "total_generated": 0
            })
    
    # Summary table
    _display_summary(all_stats)
    
    return all_stats


def _display_summary(all_stats: List[Dict[str, Any]]):
    """Display a summary table of QA generation results."""
    if not all_stats:
        return
    
    table = Table(title="\n📊 QA Generation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Chunk Config", style="cyan", no_wrap=False)
    table.add_column("Chunks", justify="right", style="dim")
    table.add_column("Factual", justify="right", style="cyan")
    table.add_column("Analytical", justify="right", style="magenta")
    table.add_column("Multi-hop", justify="right", style="green")
    table.add_column("Boundary", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="bold green")
    
    total_questions = 0
    for stats in all_stats:
        if "error" in stats:
            table.add_row(
                stats["chunk_config"],
                "—", "—", "—", "—", "—",
                f"[red]Error[/red]"
            )
        else:
            gen = stats["generated"]
            total = stats["total_generated"]
            total_questions += total
            table.add_row(
                stats["chunk_config"],
                str(stats["total_chunks"]),
                str(gen.get("factual", 0)),
                str(gen.get("analytical", 0)),
                str(gen.get("multi-hop", 0)),
                str(gen.get("boundary", 0)),
                str(total)
            )
    
    console.print(table)
    console.print(f"\n[bold green]✓ Total questions generated: {total_questions}[/bold green]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs from chunked documents")
    parser.add_argument("--chunk-file", type=Path, help="Single chunk file to process")
    parser.add_argument("--chunk-dir", type=Path, help="Directory of chunk files (batch mode)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa"),
                       help="Output directory for QA files (default: data/qa)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--num-factual", type=int, default=5,
                       help="Number of factual questions (default: 5)")
    parser.add_argument("--num-analytical", type=int, default=3,
                       help="Number of analytical questions (default: 3)")
    parser.add_argument("--num-multi-hop", type=int, default=2,
                       help="Number of multi-hop questions (default: 2)")
    parser.add_argument("--num-boundary", type=int, default=2,
                       help="Number of boundary questions (default: 2)")
    
    args = parser.parse_args()
    
    if not args.chunk_file and not args.chunk_dir:
        parser.error("Must specify either --chunk-file or --chunk-dir")
    
    try:
        if args.chunk_file:
            # Single file mode
            output_file = args.output_dir / f"{args.chunk_file.stem}__qa.jsonl"
            stats = generate_qa_for_chunks(
                chunk_file=args.chunk_file,
                output_file=output_file,
                model_name=args.model,
                num_factual=args.num_factual,
                num_analytical=args.num_analytical,
                num_multi_hop=args.num_multi_hop,
                num_boundary=args.num_boundary
            )
            console.print(f"\n[bold green]✓ Success![/bold green] Generated {stats['total_generated']} questions")
            console.print(f"Output: [cyan]{stats['output_file']}[/cyan]")
        else:
            # Batch mode
            all_stats = generate_qa_grid(
                chunk_dir=args.chunk_dir,
                output_dir=args.output_dir,
                model_name=args.model,
                num_factual=args.num_factual,
                num_analytical=args.num_analytical,
                num_multi_hop=args.num_multi_hop,
                num_boundary=args.num_boundary
            )
            
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)
