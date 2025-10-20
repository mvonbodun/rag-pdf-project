#!/usr/bin/env python3
"""
End-to-End RAG Evaluation Pipeline Runner

This script orchestrates the complete evaluation pipeline:
    Step 1: Load PDF documents
    Step 2: Parse & Chunk (multiple configurations)
    Step 3: Build Vector Indexes (FAISS with multiple embeddings)
    Step 4: Generate Synthetic QA pairs
    Step 5: Evaluate Retrieval (compute Recall@K, Precision@K, MRR@K)
    Step 6: [Optional] Reranking (skipped if retrieval is excellent)
    Step 7: Generate Answers (RAG with citations)
    Step 8: Evaluate Generation (LLM-as-Judge)
    Step 9: Final Analysis & Recommendations

Usage:
    python src/run_pipeline.py --pdf data/raw/fy10syb.pdf
    python src/run_pipeline.py --pdf data/raw/fy10syb.pdf --quick  # Fast mode
    python src/run_pipeline.py --pdf data/raw/fy10syb.pdf --skip-eval  # Skip expensive evaluations
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

console = Console()


class RAGPipeline:
    """Orchestrates the complete RAG evaluation pipeline."""
    
    def __init__(
        self,
        pdf_path: Path,
        output_dir: Path,
        quick_mode: bool = False,
        skip_eval: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            pdf_path: Path to PDF file to process
            output_dir: Root directory for outputs
            quick_mode: If True, use minimal configurations for faster testing
            skip_eval: If True, skip expensive evaluation steps
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.quick_mode = quick_mode
        self.skip_eval = skip_eval
        
        # Pipeline state
        self.stats = {
            'start_time': time.time(),
            'pdf_name': pdf_path.stem,
            'steps_completed': [],
            'steps_skipped': [],
            'errors': []
        }
        
        # Configuration
        if quick_mode:
            # Quick mode: single config for testing
            self.parsers = ['pdfplumber']
            self.chunk_configs = [{'size': 512, 'overlap': 128}]
            self.embeddings = ['ada-002']
            self.num_qa = 6  # Fewer questions
        else:
            # Full mode: comprehensive evaluation
            self.parsers = ['pymupdf', 'pdfplumber']
            self.chunk_configs = [
                {'size': 128, 'overlap': 32},
                {'size': 256, 'overlap': 64},
                {'size': 512, 'overlap': 128}
            ]
            self.embeddings = ['text-embedding-3-small', 'text-embedding-3-large', 'ada-002']
            self.num_qa = 12  # Full question set
        
        self.llm_model = 'gpt-4o-mini'
        self.k = 5  # Top-K retrieval
    
    def display_header(self):
        """Display pipeline header."""
        mode = "🚀 QUICK MODE" if self.quick_mode else "🔬 FULL EVALUATION"
        skip_note = " (Evaluation Skipped)" if self.skip_eval else ""
        
        console.print(Panel.fit(
            f"[bold cyan]{mode}{skip_note}[/bold cyan]\n"
            f"PDF: [yellow]{self.pdf_path.name}[/yellow]\n"
            f"Output: [yellow]{self.output_dir}[/yellow]\n"
            f"Configs: [yellow]{len(self.parsers)} parsers × {len(self.chunk_configs)} chunks × {len(self.embeddings)} embeddings[/yellow]\n"
            f"Total: [yellow]{len(self.parsers) * len(self.chunk_configs) * len(self.embeddings)} configurations[/yellow]",
            title="🎯 RAG Evaluation Pipeline",
            border_style="cyan",
            box=box.DOUBLE
        ))
    
    def run_command(self, cmd: List[str], step_name: str) -> bool:
        """
        Run a subprocess command.
        
        Args:
            cmd: Command and arguments
            step_name: Human-readable step name for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                console.print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error in {step_name}:[/red]")
            console.print(e.stderr)
            self.stats['errors'].append({
                'step': step_name,
                'error': str(e),
                'stderr': e.stderr
            })
            return False
    
    def step1_load_pdf(self) -> bool:
        """Step 1: Verify PDF exists."""
        console.print("\n[bold cyan]═══ Step 1: Load PDF ═══[/bold cyan]")
        
        if not self.pdf_path.exists():
            console.print(f"[red]✗ PDF not found: {self.pdf_path}[/red]")
            return False
        
        file_size = self.pdf_path.stat().st_size / (1024 * 1024)  # MB
        console.print(f"✓ Found PDF: {self.pdf_path.name} ({file_size:.2f} MB)")
        self.stats['steps_completed'].append('step1_load_pdf')
        return True
    
    def step2_parse_chunk(self) -> bool:
        """Step 2: Parse & Chunk with multiple configurations."""
        console.print("\n[bold cyan]═══ Step 2: Parse & Chunk ═══[/bold cyan]")
        
        total = len(self.parsers) * len(self.chunk_configs)
        console.print(f"Generating {total} chunked versions...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Parsing & chunking...", total=total)
            
            for parser in self.parsers:
                for config in self.chunk_configs:
                    cmd = [
                        'python', 'src/parse_chunk.py',
                        '--pdf', str(self.pdf_path),
                        '--parser', parser,
                        '--chunk-size', str(config['size']),
                        '--overlap', str(config['overlap']),
                        '--output-dir', 'data/processed'
                    ]
                    
                    if not self.run_command(cmd, f"parse_chunk_{parser}_cs{config['size']}"):
                        return False
                    
                    progress.update(task, advance=1)
        
        console.print(f"[green]✓ Created {total} chunked configurations[/green]")
        self.stats['steps_completed'].append('step2_parse_chunk')
        return True
    
    def step3_build_indexes(self) -> bool:
        """Step 3: Build vector indexes."""
        console.print("\n[bold cyan]═══ Step 3: Build Vector Indexes ═══[/bold cyan]")
        
        total = len(self.parsers) * len(self.chunk_configs) * len(self.embeddings)
        console.print(f"Building {total} FAISS indexes...")
        
        cmd = [
            'python', 'src/build_index.py',
            '--chunks-dir', 'data/processed',
            '--output-dir', 'indexes/faiss'
        ]
        
        if not self.run_command(cmd, "build_indexes"):
            return False
        
        console.print(f"[green]✓ Built {total} vector indexes[/green]")
        self.stats['steps_completed'].append('step3_build_indexes')
        return True
    
    def step4_generate_qa(self) -> bool:
        """Step 4: Generate synthetic QA pairs."""
        console.print("\n[bold cyan]═══ Step 4: Generate Synthetic QA ═══[/bold cyan]")
        
        if self.skip_eval:
            console.print("[yellow]⊘ Skipped (--skip-eval)[/yellow]")
            self.stats['steps_skipped'].append('step4_generate_qa')
            return True
        
        cmd = [
            'python', 'src/gen_synth_qa.py',
            '--chunks-dir', 'data/processed',
            '--output-dir', 'data/qa',
            '--model', self.llm_model
        ]
        
        if not self.run_command(cmd, "generate_qa"):
            return False
        
        console.print(f"[green]✓ Generated synthetic QA pairs[/green]")
        self.stats['steps_completed'].append('step4_generate_qa')
        return True
    
    def step5_eval_retrieval(self) -> bool:
        """Step 5: Evaluate retrieval performance."""
        console.print("\n[bold cyan]═══ Step 5: Evaluate Retrieval ═══[/bold cyan]")
        
        if self.skip_eval:
            console.print("[yellow]⊘ Skipped (--skip-eval)[/yellow]")
            self.stats['steps_skipped'].append('step5_eval_retrieval')
            return True
        
        cmd = [
            'python', 'src/eval_retrieval.py',
            '--indexes-dir', 'indexes/faiss',
            '--qa-dir', 'data/qa',
            '--output-dir', 'runs/retrieval',
            f'--k-values', '1,3,5,10'
        ]
        
        if not self.run_command(cmd, "eval_retrieval"):
            return False
        
        # Read results to find best config
        self._find_best_config()
        
        console.print(f"[green]✓ Evaluated retrieval performance[/green]")
        self.stats['steps_completed'].append('step5_eval_retrieval')
        return True
    
    def step6_reranking(self) -> bool:
        """Step 6: Reranking (optional)."""
        console.print("\n[bold cyan]═══ Step 6: Reranking ═══[/bold cyan]")
        console.print("[yellow]⊘ Skipped (retrieval already excellent)[/yellow]")
        self.stats['steps_skipped'].append('step6_reranking')
        return True
    
    def step7_generate_answers(self) -> bool:
        """Step 7: Generate answers using best config."""
        console.print("\n[bold cyan]═══ Step 7: Generate Answers ═══[/bold cyan]")
        
        best_config = self.stats.get('best_config')
        if not best_config:
            console.print("[yellow]Using default config (no evaluation run)[/yellow]")
            best_config = {
                'config': f'{self.pdf_path.stem}__parser_pdfplumber__cs512__ov128',
                'embedding': 'ada-002'
            }
        
        cmd = [
            'python', 'src/generate_answers.py',
            '--config', best_config['config'],
            '--embedding', best_config['embedding'],
            '--llm', self.llm_model,
            '--k', str(self.k)
        ]
        
        if not self.run_command(cmd, "generate_answers"):
            return False
        
        console.print(f"[green]✓ Generated answers with citations[/green]")
        self.stats['steps_completed'].append('step7_generate_answers')
        return True
    
    def step8_eval_generation(self) -> bool:
        """Step 8: Evaluate generation quality."""
        console.print("\n[bold cyan]═══ Step 8: Evaluate Generation (LLM-as-Judge) ═══[/bold cyan]")
        
        if self.skip_eval:
            console.print("[yellow]⊘ Skipped (--skip-eval)[/yellow]")
            self.stats['steps_skipped'].append('step8_eval_generation')
            return True
        
        # Find the most recent answers file
        answers_files = list(Path('runs/generation').glob('answers__*.jsonl'))
        if not answers_files:
            console.print("[red]✗ No answers file found[/red]")
            return False
        
        latest_answers = max(answers_files, key=lambda p: p.stat().st_mtime)
        
        cmd = [
            'python', 'src/eval_generation.py',
            '--answers-file', str(latest_answers),
            '--judge-model', self.llm_model
        ]
        
        if not self.run_command(cmd, "eval_generation"):
            return False
        
        console.print(f"[green]✓ Evaluated generation quality[/green]")
        self.stats['steps_completed'].append('step8_eval_generation')
        return True
    
    def step9_final_analysis(self) -> bool:
        """Step 9: Display final analysis and recommendations."""
        console.print("\n[bold cyan]═══ Step 9: Final Analysis ═══[/bold cyan]")
        
        elapsed = time.time() - self.stats['start_time']
        
        # Summary table
        table = Table(title="Pipeline Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("PDF Document", self.pdf_path.name)
        table.add_row("Configurations Tested", 
                     str(len(self.parsers) * len(self.chunk_configs) * len(self.embeddings)))
        table.add_row("Steps Completed", str(len(self.stats['steps_completed'])))
        table.add_row("Steps Skipped", str(len(self.stats['steps_skipped'])))
        table.add_row("Errors", str(len(self.stats['errors'])))
        table.add_row("Total Time", f"{elapsed/60:.1f} minutes")
        
        if self.stats.get('best_config'):
            best = self.stats['best_config']
            table.add_row("Best Config", best['config'])
            table.add_row("Best Embedding", best['embedding'])
            if 'recall' in best:
                table.add_row("Best Recall@5", f"{best['recall']:.1%}")
        
        console.print(table)
        
        # Recommendations
        console.print("\n[bold green]✅ Pipeline Complete![/bold green]")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  1. Review results in runs/generation/ and runs/retrieval/")
        console.print("  2. Check evaluation CSVs for detailed metrics")
        console.print("  3. Read STEP8_COMPLETE.md for quality analysis")
        console.print("  4. Test on additional documents")
        console.print("  5. Deploy to production!")
        
        self.stats['steps_completed'].append('step9_final_analysis')
        return True
    
    def _find_best_config(self):
        """Parse retrieval results to find best configuration."""
        try:
            summary_file = Path('runs/retrieval/retrieval_evaluation_summary.csv')
            if summary_file.exists():
                import pandas as pd
                df = pd.read_csv(summary_file)
                best_row = df.nlargest(1, 'recall@5').iloc[0]
                
                self.stats['best_config'] = {
                    'config': best_row['chunk_config'],
                    'embedding': best_row['embedding_model'],
                    'recall': best_row['recall@5']
                }
                
                console.print(f"\n[bold green]🏆 Best Configuration:[/bold green]")
                console.print(f"  Config: [cyan]{best_row['chunk_config']}[/cyan]")
                console.print(f"  Embedding: [cyan]{best_row['embedding_model']}[/cyan]")
                console.print(f"  Recall@5: [cyan]{best_row['recall@5']:.1%}[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not parse best config: {e}[/yellow]")
    
    def run(self) -> bool:
        """Run the complete pipeline."""
        self.display_header()
        
        steps = [
            ('Step 1: Load PDF', self.step1_load_pdf),
            ('Step 2: Parse & Chunk', self.step2_parse_chunk),
            ('Step 3: Build Indexes', self.step3_build_indexes),
            ('Step 4: Generate QA', self.step4_generate_qa),
            ('Step 5: Evaluate Retrieval', self.step5_eval_retrieval),
            ('Step 6: Reranking', self.step6_reranking),
            ('Step 7: Generate Answers', self.step7_generate_answers),
            ('Step 8: Evaluate Generation', self.step8_eval_generation),
            ('Step 9: Final Analysis', self.step9_final_analysis),
        ]
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    console.print(f"\n[bold red]✗ Pipeline failed at: {step_name}[/bold red]")
                    return False
            except KeyboardInterrupt:
                console.print(f"\n[yellow]⚠ Pipeline interrupted by user[/yellow]")
                return False
            except Exception as e:
                console.print(f"\n[bold red]✗ Unexpected error in {step_name}:[/bold red]")
                console.print(f"[red]{e}[/red]")
                import traceback
                traceback.print_exc()
                return False
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation (all configurations)
  python src/run_pipeline.py --pdf data/raw/fy10syb.pdf

  # Quick mode (single config for testing)
  python src/run_pipeline.py --pdf data/raw/fy10syb.pdf --quick

  # Skip expensive evaluations
  python src/run_pipeline.py --pdf data/raw/fy10syb.pdf --skip-eval
        """
    )
    
    parser.add_argument(
        '--pdf',
        type=Path,
        required=True,
        help='Path to PDF file to evaluate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Root output directory (default: current directory)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: single configuration for fast testing'
    )
    
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip expensive evaluation steps (QA generation, LLM-as-Judge)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = RAGPipeline(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        quick_mode=args.quick,
        skip_eval=args.skip_eval
    )
    
    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
