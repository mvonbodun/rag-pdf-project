#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run chunking grid: all combinations of parsers × chunk_sizes × overlaps

This script reads the grid.default.yaml config and runs parse_chunk.py
for all combinations, creating the complete matrix of chunked files.
"""

import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import time

from src.parse_chunk import parse_and_chunk_pdf

console = Console()

def load_config(config_path: Path) -> dict:
    """Load the YAML configuration"""
    with config_path.open('r') as f:
        return yaml.safe_load(f)

def run_chunking_grid(config_path: Path = Path("configs/grid.default.yaml")):
    """
    Run the complete chunking grid
    
    Reads config and generates:
    - parsers × chunk_sizes × overlaps combinations
    - For each combination, parse and chunk all documents
    """
    # Load config
    config = load_config(config_path)
    
    # Extract parameters
    docs = config['dataset']['docs']
    parsers = config['chunking']['parsers']
    chunk_sizes = config['chunking']['chunk_sizes']
    overlaps = config['chunking']['overlaps']
    tokenizer = config['chunking'].get('tokenizer', 'cl100k_base')
    
    # Output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total combinations
    total_combinations = len(docs) * len(parsers) * len(chunk_sizes) * len(overlaps)
    
    # Display grid info
    console.print("\n[bold cyan]═══ RAG Chunking Grid ═══[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Values", style="green")
    
    table.add_row("Documents", f"{len(docs)} files")
    table.add_row("Parsers", ", ".join(parsers))
    table.add_row("Chunk Sizes", ", ".join(map(str, chunk_sizes)))
    table.add_row("Overlaps", ", ".join(map(str, overlaps)))
    table.add_row("Tokenizer", tokenizer)
    table.add_row("[bold]Total Combinations[/bold]", f"[bold yellow]{total_combinations}[/bold yellow]")
    
    console.print(table)
    console.print()
    
    # Track results
    results = []
    
    # Run grid with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Processing combinations...", total=total_combinations)
        
        for doc_path_str in docs:
            doc_path = Path(doc_path_str)
            
            if not doc_path.exists():
                console.print(f"[red]⚠ Warning: Document not found: {doc_path}[/red]")
                progress.update(task, advance=len(parsers) * len(chunk_sizes) * len(overlaps))
                continue
            
            for parser in parsers:
                for chunk_size in chunk_sizes:
                    for overlap in overlaps:
                        # Update progress description
                        desc = f"[cyan]{doc_path.name} | {parser} | cs={chunk_size} ov={overlap}"
                        progress.update(task, description=desc)
                        
                        try:
                            # Run parsing and chunking
                            start_time = time.time()
                            output_file = parse_and_chunk_pdf(
                                pdf_path=doc_path,
                                parser=parser,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                output_dir=output_dir,
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
                            console.print(f"[red]✗ Error: {e}[/red]")
                            results.append({
                                'doc': doc_path.name,
                                'parser': parser,
                                'chunk_size': chunk_size,
                                'overlap': overlap,
                                'chunks': 0,
                                'time': 0,
                                'error': str(e)
                            })
                        
                        progress.update(task, advance=1)
    
    # Display results summary
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
    console.print()
    
    # Parser comparison
    console.print("\n[bold cyan]═══ Parser Comparison (at cs=256, ov=64) ═══[/bold cyan]\n")
    
    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("Parser", style="cyan")
    comparison_table.add_column("Chunks", justify="right", style="green")
    comparison_table.add_column("Difference from PyMuPDF", justify="right", style="yellow")
    
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
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the complete chunking grid")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grid.default.yaml"),
        help="Path to grid configuration YAML"
    )
    
    args = parser.parse_args()
    
    run_chunking_grid(args.config)
