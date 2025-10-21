#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Index Building Module

Supports:
- Multiple embedding models (OpenAI, SentenceTransformers)
- Batch embedding with ThreadPoolExecutor
- FAISS index creation and persistence
- Per-configuration index storage

Index naming: {doc}__parser_{parser}__cs{cs}__ov{ov}__emb_{emb_id}
"""

import json
import numpy as np
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    # Try to load manually if dotenv not available
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

import faiss

# Embedding providers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. Install with: pip install openai")


@dataclass
class EmbeddingConfig:
    """Configuration for an embedding model"""
    id: str
    provider: str  # "openai" or "sentence-transformers"
    model: str
    dimension: int = 0  # Will be set after first embedding


class EmbeddingModel:
    """Wrapper for different embedding providers"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the embedding model based on provider"""
        if self.config.provider == "sentence-transformers":
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers not installed")
            print(f"Loading SentenceTransformer model: {self.config.model}")
            # Disable multiprocessing to avoid segfault on macOS
            self.model = SentenceTransformer(self.config.model, device='cpu')
            # Get dimension from model
            self.config.dimension = self.model.get_sentence_embedding_dimension()
            
        elif self.config.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai not installed")
            print(f"Using OpenAI model: {self.config.model}")
            # Check for API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            # OpenAI dimensions (from their docs)
            if "text-embedding-3-small" in self.config.model:
                self.config.dimension = 1536
            elif "text-embedding-3-large" in self.config.model:
                self.config.dimension = 3072
            elif "text-embedding-ada-002" in self.config.model:
                self.config.dimension = 1536
            else:
                self.config.dimension = 1536  # default
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if self.config.provider == "sentence-transformers":
            return self._embed_sentence_transformers(texts, batch_size)
        elif self.config.provider == "openai":
            return self._embed_openai(texts, batch_size)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _embed_sentence_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed using SentenceTransformers"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            device='cpu'  # Force CPU to avoid multiprocessing issues
        )
        return embeddings
    
    def _embed_openai(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Embed using OpenAI API"""
        embeddings = []
        
        # Process in batches to respect rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    model=self.config.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limits
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error embedding batch {i}-{i+batch_size}: {e}")
                # Return zeros for failed batch
                embeddings.extend([[0.0] * self.config.dimension] * len(batch))
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        
        return embeddings


def build_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Build a FAISS index from embeddings
    
    Args:
        embeddings: numpy array of shape (n_chunks, dimension)
        index_type: Type of FAISS index ("flat", "ivf", "hnsw")
        
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    n_chunks = embeddings.shape[0]
    
    if index_type == "flat":
        # Simple flat index (exact search, best for small datasets)
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine with normalized vectors)
        index.add(embeddings)
        
    elif index_type == "ivf":
        # IVF index (faster for large datasets)
        n_clusters = min(100, n_chunks // 10)  # Heuristic
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        index.train(embeddings)
        index.add(embeddings)
        
    elif index_type == "hnsw":
        # HNSW index (graph-based, very fast)
        M = 32  # Number of connections per layer
        index = faiss.IndexHNSWFlat(dimension, M)
        index.add(embeddings)
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    print(f"Built {index_type} FAISS index: {n_chunks} vectors, {dimension} dimensions")
    return index


def load_chunks(chunk_file: Path) -> Tuple[List[str], List[Dict]]:
    """
    Load chunks from JSONL file
    
    Returns:
        texts: List of chunk texts
        chunks: List of full chunk dictionaries
    """
    chunks = []
    with chunk_file.open('r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    texts = [chunk['text'] for chunk in chunks]
    return texts, chunks


def save_index(
    index: faiss.Index,
    chunk_file: Path,
    chunks: List[Dict],
    embedding_config: EmbeddingConfig,
    output_dir: Path
):
    """
    Save FAISS index and metadata
    
    Directory structure:
    output_dir/
        index.faiss          # FAISS index file
        chunks.jsonl         # Original chunks
        metadata.json        # Index metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    
    # Save chunks
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open('w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    # Save metadata
    metadata = {
        "source_file": str(chunk_file),
        "n_chunks": len(chunks),
        "embedding_model": {
            "id": embedding_config.id,
            "provider": embedding_config.provider,
            "model": embedding_config.model,
            "dimension": embedding_config.dimension
        },
        "index_type": "flat",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved index to {output_dir}")
    print(f"  - {index_path.name}: FAISS index")
    print(f"  - {chunks_path.name}: {len(chunks)} chunks")
    print(f"  - {metadata_path.name}: metadata")


def build_index_for_chunk_file(
    chunk_file: Path,
    embedding_config: EmbeddingConfig,
    output_base_dir: Path,
    batch_size: int = 32,
    index_type: str = "flat"
) -> Path:
    """
    Build and save a FAISS index for a chunk file
    
    Args:
        chunk_file: Path to chunks JSONL file
        embedding_config: Embedding model configuration
        output_base_dir: Base directory for index storage
        batch_size: Batch size for embedding
        index_type: Type of FAISS index
        
    Returns:
        Path to index directory
    """
    print(f"\n{'='*60}")
    print(f"Building index: {chunk_file.name}")
    print(f"Embedding model: {embedding_config.id} ({embedding_config.model})")
    print(f"{'='*60}")
    
    # Load chunks
    print("Loading chunks...")
    texts, chunks = load_chunks(chunk_file)
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(embedding_config)
    
    # Embed chunks
    print(f"Embedding {len(texts)} chunks...")
    start_time = time.time()
    embeddings = embedding_model.embed_batch(texts, batch_size)
    elapsed = time.time() - start_time
    print(f"Embedded in {elapsed:.2f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    
    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings, index_type)
    
    # Determine output directory name
    # Format: {doc}__parser_{parser}__cs{cs}__ov{ov}__emb_{emb_id}
    chunk_stem = chunk_file.stem  # e.g., "fy10syb__parser_pymupdf__cs256__ov64"
    index_dir_name = f"{chunk_stem}__emb_{embedding_config.id}"
    output_dir = output_base_dir / index_dir_name
    
    # Save index
    save_index(index, chunk_file, chunks, embedding_config, output_dir)
    
    return output_dir


def build_index_grid(
    chunk_dir: Path,
    embedding_configs: List[EmbeddingConfig],
    output_base_dir: Path,
    batch_size: int = 32,
    index_type: str = "flat",
    pattern: str = "*.jsonl",
    prefix: str = None
) -> List[Dict]:
    """
    Build indexes for all chunk files and embedding models
    
    Args:
        chunk_dir: Directory containing chunk JSONL files
        embedding_configs: List of embedding configurations
        output_base_dir: Base directory for index storage
        batch_size: Batch size for embedding
        index_type: Type of FAISS index
        pattern: Glob pattern for chunk files
        prefix: Only process files starting with this prefix
        
    Returns:
        List of results with index paths and metadata
    """
    # Find all chunk files
    chunk_files = sorted(chunk_dir.glob(pattern))
    
    # Filter out non-parser files (like sample.jsonl)
    chunk_files = [f for f in chunk_files if "__parser_" in f.name]
    
    # Filter by prefix if provided
    if prefix:
        chunk_files = [f for f in chunk_files if f.stem.startswith(prefix)]
    
    if not chunk_files:
        print(f"No chunk files found in {chunk_dir} matching {pattern}")
        return []
    
    print(f"\nFound {len(chunk_files)} chunk files")
    print(f"Using {len(embedding_configs)} embedding models")
    print(f"Total indexes to build: {len(chunk_files) * len(embedding_configs)}")
    
    results = []
    total = len(chunk_files) * len(embedding_configs)
    current = 0
    
    for chunk_file in chunk_files:
        for emb_config in embedding_configs:
            current += 1
            print(f"\n[{current}/{total}] Processing...")
            
            try:
                start_time = time.time()
                output_dir = build_index_for_chunk_file(
                    chunk_file=chunk_file,
                    embedding_config=emb_config,
                    output_base_dir=output_base_dir,
                    batch_size=batch_size,
                    index_type=index_type
                )
                elapsed = time.time() - start_time
                
                results.append({
                    "chunk_file": chunk_file.name,
                    "embedding_model": emb_config.id,
                    "index_dir": output_dir.name,
                    "status": "success",
                    "time": elapsed
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "chunk_file": chunk_file.name,
                    "embedding_model": emb_config.id,
                    "status": "failed",
                    "error": str(e)
                })
    
    return results


def main():
    """Example usage"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Build FAISS indexes from chunks")
    parser.add_argument(
        "--chunk-file",
        type=Path,
        help="Path to a single chunk file to index"
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing chunk files (for batch mode)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/providers.yaml"),
        help="Path to providers config"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("indexes/faiss"),
        help="Output directory for indexes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="flat",
        help="Type of FAISS index"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Only process chunk files starting with this prefix (e.g., 'fy10syb')"
    )
    
    args = parser.parse_args()
    
    # Load embedding configs
    with args.config.open('r') as f:
        config = yaml.safe_load(f)
    
    embedding_configs = [
        EmbeddingConfig(
            id=emb['id'],
            provider=emb['provider'],
            model=emb['model']
        )
        for emb in config['embeddings']
    ]
    
    if args.chunk_file:
        # Single file mode
        for emb_config in embedding_configs:
            build_index_for_chunk_file(
                chunk_file=args.chunk_file,
                embedding_config=emb_config,
                output_base_dir=args.output_dir,
                batch_size=args.batch_size,
                index_type=args.index_type
            )
    else:
        # Batch mode
        results = build_index_grid(
            chunk_dir=args.chunk_dir,
            embedding_configs=embedding_configs,
            output_base_dir=args.output_dir,
            batch_size=args.batch_size,
            index_type=args.index_type,
            prefix=args.prefix
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("INDEX BUILDING SUMMARY")
        print(f"{'='*60}")
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        total_time = sum(r.get('time', 0) for r in results if 'time' in r)
        
        print(f"Total: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        if successful > 0:
            print(f"Average time: {total_time/successful:.2f}s per index")


if __name__ == "__main__":
    main()


