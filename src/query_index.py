#!/usr/bin/env python3
"""
Query a FAISS vector index with natural language queries.

This script allows you to interactively search a FAISS index by providing:
1. The index directory path
2. A natural language query
3. Number of results to return (top-K)

The script will embed your query using the same embedding model that was used
to build the index, search for similar chunks, and display the results in a
readable format.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

console = Console()


class QueryEmbedder:
    """Embed queries using the same model as the index."""
    
    def __init__(self, provider: str, model: str):
        """
        Initialize the query embedder.
        
        Args:
            provider: Embedding provider ('openai' or 'sentence-transformers')
            model: Model name/identifier
        """
        self.provider = provider
        self.model_name = model
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the embedding model."""
        if self.provider == "sentence-transformers":
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            console.print(f"[dim]Loading SentenceTransformer model: {self.model_name}[/dim]")
            self.model = SentenceTransformer(self.model_name, device='cpu')
            
        elif self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError(
                    "openai not installed. "
                    "Install with: pip install openai"
                )
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it before using OpenAI embeddings."
                )
            console.print(f"[dim]Using OpenAI model: {self.model_name}[/dim]")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def embed(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Args:
            query: The query text to embed
            
        Returns:
            Normalized embedding vector as numpy array
        """
        if self.provider == "sentence-transformers":
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                device='cpu'
            )[0]
            return embedding
            
        elif self.provider == "openai":
            response = openai.embeddings.create(
                model=self.model_name,
                input=[query]
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


class FAISSIndexQuery:
    """Query a FAISS vector index."""
    
    def __init__(self, index_dir: Path):
        """
        Load a FAISS index with its metadata and chunks.
        
        Args:
            index_dir: Directory containing index.faiss, chunks.jsonl, metadata.json
        """
        self.index_dir = index_dir
        
        # Validate directory exists
        if not index_dir.exists():
            raise ValueError(f"Index directory does not exist: {index_dir}")
        
        # Load metadata first to get embedding info
        metadata_path = index_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"metadata.json not found in {index_dir}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load FAISS index
        index_path = index_dir / "index.faiss"
        if not index_path.exists():
            raise ValueError(f"index.faiss not found in {index_dir}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise ValueError(f"chunks.jsonl not found in {index_dir}")
        
        self.chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        console.print(f"[green]✓[/green] Loaded index: [cyan]{index_dir.name}[/cyan]")
        console.print(f"  Vectors: {self.index.ntotal}")
        console.print(f"  Dimensions: {self.index.d}")
        console.print(f"  Chunks: {len(self.chunks)}")
    
    def get_embedding_info(self) -> Tuple[str, str]:
        """
        Extract embedding provider and model from metadata.
        
        Returns:
            Tuple of (provider, model_name)
        """
        emb_meta = self.metadata['embedding_model']
        
        if isinstance(emb_meta, dict):
            provider = emb_meta.get('provider', 'openai')
            model_name = emb_meta.get('model', emb_meta.get('id', 'text-embedding-3-small'))
        else:
            # Legacy format - assume it's an ID like "openai-small"
            if 'openai' in emb_meta or 'ada' in emb_meta or 'text-embedding' in emb_meta:
                provider = 'openai'
                # Map common IDs to full model names
                if 'small' in emb_meta:
                    model_name = 'text-embedding-3-small'
                elif 'large' in emb_meta:
                    model_name = 'text-embedding-3-large'
                elif 'ada' in emb_meta:
                    model_name = 'text-embedding-ada-002'
                else:
                    model_name = emb_meta
            else:
                provider = 'sentence-transformers'
                model_name = emb_meta
        
        return provider, model_name
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        """
        Search the index for top-K nearest neighbors.
        
        Args:
            query_embedding: Query vector (normalized)
            k: Number of results to return
        
        Returns:
            List of result dictionaries with chunk data and scores
        """
        # Ensure query is 2D with shape (1, dim)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['search_score'] = float(score)
                chunk['index_position'] = int(idx)
                results.append(chunk)
        
        return results


def list_available_indexes(base_dir: Path = Path("indexes/faiss")) -> List[Path]:
    """
    List all available FAISS indexes.
    
    Args:
        base_dir: Base directory containing FAISS indexes
        
    Returns:
        List of index directory paths
    """
    if not base_dir.exists():
        return []
    
    indexes = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "index.faiss").exists():
            indexes.append(item)
    
    return sorted(indexes)


def select_index_interactive(base_dir: Path = Path("indexes/faiss")) -> Path:
    """
    Let user select an index from available options.
    
    Args:
        base_dir: Base directory containing FAISS indexes
        
    Returns:
        Selected index directory path
    """
    indexes = list_available_indexes(base_dir)
    
    if not indexes:
        console.print(f"[red]No FAISS indexes found in {base_dir}[/red]")
        sys.exit(1)
    
    # Group indexes by PDF name for better display
    from collections import defaultdict
    grouped = defaultdict(list)
    for idx in indexes:
        # Extract PDF name (everything before __parser_)
        name = idx.name
        pdf_name = name.split("__parser_")[0] if "__parser_" in name else name
        grouped[pdf_name].append(idx)
    
    console.print(Panel.fit(
        "[bold cyan]Select a FAISS Index[/bold cyan]\n"
        f"Found {len(indexes)} indexes across {len(grouped)} documents",
        title="📚 Available Indexes",
        border_style="cyan"
    ))
    
    # Display grouped indexes
    choice_map = {}
    choice_num = 1
    
    for pdf_name in sorted(grouped.keys()):
        console.print(f"\n[bold yellow]{pdf_name}[/bold yellow]")
        for idx_path in grouped[pdf_name]:
            # Extract config details
            name = idx_path.name
            parts = name.split("__")
            
            # Format display string
            if len(parts) >= 4:
                parser = parts[1].replace("parser_", "")
                chunk_info = parts[2] + "_" + parts[3]
                emb_info = parts[4].replace("emb_", "") if len(parts) > 4 else "unknown"
                display = f"  [{choice_num:2d}] {parser:10s} | {chunk_info:15s} | {emb_info}"
            else:
                display = f"  [{choice_num:2d}] {name}"
            
            console.print(display)
            choice_map[choice_num] = idx_path
            choice_num += 1
    
    # Get user choice
    while True:
        try:
            choice_str = console.input(f"\n[bold cyan]Select index (1-{len(indexes)}) or 'q' to quit:[/bold cyan] ").strip()
            
            if choice_str.lower() in ['q', 'quit', 'exit']:
                console.print("[dim]Goodbye![/dim]")
                sys.exit(0)
            
            choice = int(choice_str)
            if choice in choice_map:
                selected = choice_map[choice]
                console.print(f"[green]✓[/green] Selected: [cyan]{selected.name}[/cyan]\n")
                return selected
            else:
                console.print(f"[red]Invalid choice. Please enter a number between 1 and {len(indexes)}[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            sys.exit(0)


def display_results(query: str, results: List[dict], index_name: str):
    """
    Display search results in a beautiful, readable format.
    
    Args:
        query: The original query string
        results: List of result dictionaries
        index_name: Name of the index that was searched
    """
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Query:[/bold cyan] {query}\n"
        f"[bold cyan]Index:[/bold cyan] {index_name}\n"
        f"[bold cyan]Results:[/bold cyan] {len(results)}",
        title="🔍 Search Results",
        border_style="cyan"
    ))
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    # Display each result
    for i, result in enumerate(results, 1):
        # Create result panel
        score = result.get('search_score', 0.0)
        chunk_id = result.get('id', 'unknown')
        text = result.get('text', '')
        metadata = result.get('metadata', {})
        
        # Build metadata string
        meta_parts = []
        if 'page' in metadata:
            meta_parts.append(f"Page {metadata['page']}")
        if 'chunk_index' in metadata:
            meta_parts.append(f"Chunk {metadata['chunk_index']}")
        meta_str = " • ".join(meta_parts) if meta_parts else "No metadata"
        
        # Color code by score
        if score > 0.85:
            score_color = "green"
        elif score > 0.75:
            score_color = "yellow"
        else:
            score_color = "red"
        
        # Format text with word wrap
        text_display = text[:500] + "..." if len(text) > 500 else text
        
        console.print(f"\n[bold]Result {i}[/bold]")
        console.print(f"[{score_color}]Score: {score:.4f}[/{score_color}] • [dim]{meta_str}[/dim]")
        console.print(Panel(
            text_display,
            title=f"[dim]{chunk_id}[/dim]",
            border_style="dim",
            box=box.ROUNDED
        ))


def interactive_mode(index_dir: Path = None, default_k: int = 5):
    """
    Run in interactive mode, allowing multiple queries.
    
    Args:
        index_dir: Path to the FAISS index directory (None to prompt for selection)
        default_k: Default number of results to return
    """
    console.print(Panel.fit(
        "[bold cyan]Interactive Query Mode[/bold cyan]\n"
        "Enter queries to search the index.\n"
        "Commands: 'quit'/'exit' to stop, 'k=N' to change result count, 'switch' to change index",
        title="🔍 FAISS Index Query Tool",
        border_style="cyan"
    ))
    
    # Select index if not provided
    if index_dir is None:
        index_dir = select_index_interactive()
    
    # Load index
    try:
        index_query = FAISSIndexQuery(index_dir)
    except Exception as e:
        console.print(f"[red]Error loading index: {e}[/red]")
        return
    
    # Initialize embedder
    try:
        provider, model_name = index_query.get_embedding_info()
        embedder = QueryEmbedder(provider, model_name)
    except Exception as e:
        console.print(f"[red]Error initializing embedder: {e}[/red]")
        return
    
    console.print()
    k = default_k
    
    while True:
        try:
            # Get query from user
            query = console.input(f"\n[bold cyan]Query (k={k}):[/bold cyan] ").strip()
            
            if not query:
                continue
            
            # Check for commands
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("[dim]Goodbye![/dim]")
                break
            
            # Check for switch index command
            if query.lower() in ['switch', 'change', 'index']:
                console.print()
                new_index_dir = select_index_interactive()
                
                # Reload index
                try:
                    index_query = FAISSIndexQuery(new_index_dir)
                    provider, model_name = index_query.get_embedding_info()
                    embedder = QueryEmbedder(provider, model_name)
                    index_dir = new_index_dir
                    console.print(f"[green]✓[/green] Switched to: [cyan]{index_dir.name}[/cyan]\n")
                except Exception as e:
                    console.print(f"[red]Error loading new index: {e}[/red]")
                    console.print("[yellow]Keeping current index[/yellow]")
                continue
            
            # Check for k adjustment
            if query.lower().startswith('k='):
                try:
                    k = int(query[2:])
                    console.print(f"[green]✓[/green] Set k={k}")
                    continue
                except ValueError:
                    console.print("[red]Invalid k value. Use format: k=5[/red]")
                    continue
            
            # Check for help command
            if query.lower() in ['help', '?']:
                console.print(Panel(
                    "[bold]Available Commands:[/bold]\n\n"
                    "[cyan]switch[/cyan] - Change to a different index\n"
                    "[cyan]k=N[/cyan] - Set number of results to N\n"
                    "[cyan]quit[/cyan] - Exit the tool\n"
                    "[cyan]help[/cyan] - Show this help message\n\n"
                    "Or just enter a query to search!",
                    title="Help",
                    border_style="blue"
                ))
                continue
            
            # Embed query
            console.print("[dim]Embedding query...[/dim]")
            query_embedding = embedder.embed(query)
            
            # Search
            console.print("[dim]Searching index...[/dim]")
            results = index_query.search(query_embedding, k=k)
            
            # Display results
            display_results(query, results, index_dir.name)
            
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'quit' to exit or continue querying.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


def single_query_mode(index_dir: Path, query: str, k: int = 5):
    """
    Run a single query and exit.
    
    Args:
        index_dir: Path to the FAISS index directory
        query: The query string
        k: Number of results to return
    """
    try:
        # Load index
        index_query = FAISSIndexQuery(index_dir)
        
        # Initialize embedder
        provider, model_name = index_query.get_embedding_info()
        embedder = QueryEmbedder(provider, model_name)
        
        # Embed query
        console.print("[dim]Embedding query...[/dim]")
        query_embedding = embedder.embed(query)
        
        # Search
        console.print("[dim]Searching index...[/dim]")
        results = index_query.search(query_embedding, k=k)
        
        # Display results
        display_results(query, results, index_dir.name)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query a FAISS vector index with natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with index selection menu
  python src/query_index.py
  
  # Interactive mode with specific index
  python src/query_index.py --index indexes/faiss/fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small
  
  # Single query
  python src/query_index.py --index indexes/faiss/fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \\
      --query "What is the total budget?" --k 3
  
  # Short form with index name only (assumes indexes/faiss/ prefix)
  python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \\
      -q "What is the total budget?" -k 3
      
Interactive Commands:
  switch       - Change to a different index
  k=N          - Set number of results to N
  help or ?    - Show help message
  quit or exit - Exit the tool
        """
    )
    
    parser.add_argument(
        "-i", "--index",
        type=str,
        help="Path to FAISS index directory (optional - will prompt if omitted). "
             "Can be full path like 'indexes/faiss/myindex__emb_openai-small' or short name 'myindex__emb_openai-small'"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Query string (if omitted, runs in interactive mode)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Parse index path if provided
    index_path = None
    if args.index:
        index_path = Path(args.index)
        if not index_path.exists():
            # Try assuming it's relative to indexes/faiss/
            index_path = Path("indexes/faiss") / args.index
            if not index_path.exists():
                console.print(f"[red]Error: Index directory not found: {args.index}[/red]")
                console.print(f"[dim]Tried: {args.index} and indexes/faiss/{args.index}[/dim]")
                sys.exit(1)
    
    # Run in appropriate mode
    if args.query:
        # Single query mode requires an index
        if not index_path:
            console.print("[red]Error: --index is required when using --query[/red]")
            sys.exit(1)
        single_query_mode(index_path, args.query, args.top_k)
    else:
        # Interactive mode - index is optional (will prompt if not provided)
        interactive_mode(index_path, args.top_k)


if __name__ == "__main__":
    main()
