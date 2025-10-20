#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Parsing & Chunking Module

Supports multiple parsing strategies:
1. PyMuPDF (fitz): Fast, text extraction with basic structure
2. unstructured: Layout-aware with semantic blocks (headings, paragraphs, tables)
3. pdfplumber: Excellent for tables and detailed layout

Each parser extracts structured blocks, then chunks are created with:
- Token-based sizing (using tiktoken)
- Configurable overlap
- Preservation of semantic units where possible
"""

import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Literal, Tuple, Optional
from dataclasses import dataclass, field

import tiktoken

# Parser imports (with fallback handling)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")

try:
    from unstructured.partition.pdf import partition_pdf
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False
    print("Warning: unstructured not installed. Install with: pip install unstructured")

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")


ParserType = Literal["pymupdf", "unstructured", "pdfplumber"]


@dataclass
class Block:
    """Represents a semantic block from PDF (paragraph, heading, table, etc)"""
    type: str  # "paragraph", "heading", "list", "table", etc
    text: str
    page_num: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a token-sized chunk with overlap"""
    id: str
    text: str
    tokens: List[int]  # token IDs from tiktoken
    token_count: int
    source_blocks: List[int]  # indices of blocks this chunk came from
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Export to JSON-serializable format"""
        return {
            "id": self.id,
            "text": self.text,
            "token_count": self.token_count,
            "source_blocks": self.source_blocks,
            "metadata": self.metadata
        }


class PDFParser:
    """Multi-strategy PDF parser with chunking capabilities"""
    
    def __init__(self, parser: ParserType = "pymupdf", tokenizer: str = "cl100k_base"):
        """
        Initialize parser
        
        Args:
            parser: Which parsing strategy to use
            tokenizer: tiktoken encoding name (cl100k_base for GPT-4, GPT-3.5)
        """
        self.parser = parser
        self.encoding = tiktoken.get_encoding(tokenizer)
        
        # Validate parser availability
        if parser == "pymupdf" and not HAS_PYMUPDF:
            raise ImportError("PyMuPDF not available")
        elif parser == "unstructured" and not HAS_UNSTRUCTURED:
            raise ImportError("unstructured not available")
        elif parser == "pdfplumber" and not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber not available")
    
    def parse(self, pdf_path: Path) -> List[Block]:
        """
        Parse PDF into structured blocks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Block objects with semantic structure
        """
        if self.parser == "pymupdf":
            return self._parse_pymupdf(pdf_path)
        elif self.parser == "unstructured":
            return self._parse_unstructured(pdf_path)
        elif self.parser == "pdfplumber":
            return self._parse_pdfplumber(pdf_path)
        else:
            raise ValueError(f"Unknown parser: {self.parser}")
    
    def _parse_pymupdf(self, pdf_path: Path) -> List[Block]:
        """
        Parse using PyMuPDF (fitz) - fast, basic structure
        
        Strategy:
        - Extract text blocks per page
        - Identify headings by font size heuristics
        - Split paragraphs by blank lines
        """
        blocks = []
        
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                # Get text with layout preservation
                text = page.get_text("text")
                
                # Split into paragraphs (simple heuristic: double newlines)
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                
                for para in paragraphs:
                    # Heuristic: short lines with title case might be headings
                    lines = para.split("\n")
                    if len(lines) == 1 and len(para) < 100 and para[0].isupper():
                        block_type = "heading"
                    else:
                        block_type = "paragraph"
                    
                    blocks.append(Block(
                        type=block_type,
                        text=para,
                        page_num=page_num + 1,
                        metadata={"parser": "pymupdf"}
                    ))
        
        return blocks
    
    def _parse_unstructured(self, pdf_path: Path) -> List[Block]:
        """
        Parse using unstructured library - layout-aware, semantic blocks
        
        Strategy:
        - Use unstructured's built-in layout detection
        - Preserves document structure (titles, headers, lists, tables)
        - Best for complex documents with varied formatting
        """
        blocks = []
        
        # Partition PDF with unstructured
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",  # or "fast" for speed vs accuracy trade-off
            infer_table_structure=True,
            include_page_breaks=True
        )
        
        for elem in elements:
            # Map unstructured element types to our Block types
            elem_type = elem.category.lower()
            
            # Get page number if available
            page_num = getattr(elem.metadata, 'page_number', 0) if hasattr(elem, 'metadata') else 0
            
            blocks.append(Block(
                type=elem_type,
                text=str(elem),
                page_num=page_num,
                metadata={
                    "parser": "unstructured",
                    "category": elem.category
                }
            ))
        
        return blocks
    
    def _parse_pdfplumber(self, pdf_path: Path) -> List[Block]:
        """
        Parse using pdfplumber - excellent for tables and precise layout
        
        Strategy:
        - Extract text with layout preservation
        - Detect and extract tables separately
        - Good for documents with tabular data
        """
        blocks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables first
                tables = page.extract_tables()
                for table in tables:
                    # Convert table to text representation
                    table_text = "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in table])
                    blocks.append(Block(
                        type="table",
                        text=table_text,
                        page_num=page_num + 1,
                        metadata={"parser": "pdfplumber", "rows": len(table)}
                    ))
                
                # Extract regular text (excluding table areas)
                text = page.extract_text()
                if text:
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    
                    for para in paragraphs:
                        # Simple heading detection
                        if len(para) < 100 and para.isupper():
                            block_type = "heading"
                        elif para.startswith("•") or para.startswith("-"):
                            block_type = "list_item"
                        else:
                            block_type = "paragraph"
                        
                        blocks.append(Block(
                            type=block_type,
                            text=para,
                            page_num=page_num + 1,
                            metadata={"parser": "pdfplumber"}
                        ))
        
        return blocks
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using tiktoken"""
        tokens = self.encoding.encode(text)
        return [self.encoding.decode([t]) for t in tokens]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def create_chunks(
        self,
        blocks: List[Block],
        chunk_size: int,
        overlap: int,
        doc_name: str
    ) -> List[Chunk]:
        """
        Create token-constrained chunks with overlap from blocks
        
        Strategy:
        1. Start with semantic blocks (unit of meaning)
        2. If block > chunk_size, split it with overlap
        3. If block < chunk_size, try to combine with neighbors
        4. Ensure token-based constraints
        
        Args:
            blocks: List of parsed blocks
            chunk_size: Maximum tokens per chunk
            overlap: Overlap tokens between chunks
            doc_name: Document name for chunk IDs
            
        Returns:
            List of Chunk objects
        """
        # Validate parameters
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        
        chunks = []
        current_tokens = []
        current_text = []
        current_blocks = []
        chunk_counter = 0
        
        for block_idx, block in enumerate(blocks):
            block_tokens = self.encoding.encode(block.text)
            
            # Case 1: Block alone exceeds chunk_size - split it with overlap
            if len(block_tokens) > chunk_size:
                # Flush any accumulated content first
                if current_tokens:
                    chunks.append(self._make_chunk(
                        current_tokens,
                        current_text,
                        current_blocks,
                        chunk_counter,
                        doc_name
                    ))
                    chunk_counter += 1
                    current_tokens = []
                    current_text = []
                    current_blocks = []
                
                # Split large block with overlap
                sub_chunks = self._split_block_with_overlap(
                    block_tokens,
                    block.text,
                    chunk_size,
                    overlap
                )
                
                for sub_text, sub_tokens in sub_chunks:
                    chunks.append(self._make_chunk(
                        sub_tokens,
                        [sub_text],
                        [block_idx],
                        chunk_counter,
                        doc_name
                    ))
                    chunk_counter += 1
                
            # Case 2: Adding block would exceed chunk_size - start new chunk with overlap
            elif len(current_tokens) + len(block_tokens) > chunk_size:
                # Create chunk from accumulated content
                chunks.append(self._make_chunk(
                    current_tokens,
                    current_text,
                    current_blocks,
                    chunk_counter,
                    doc_name
                ))
                chunk_counter += 1
                
                # Start new chunk with overlap from previous
                if overlap > 0 and current_tokens:
                    overlap_tokens = current_tokens[-overlap:]
                    overlap_text = self.encoding.decode(overlap_tokens)
                    current_tokens = overlap_tokens
                    current_text = [overlap_text]
                    current_blocks = current_blocks[-1:] if current_blocks else []
                else:
                    current_tokens = []
                    current_text = []
                    current_blocks = []
                
                # Add current block
                current_tokens.extend(block_tokens)
                current_text.append(block.text)
                current_blocks.append(block_idx)
            
            # Case 3: Block fits in current chunk
            else:
                current_tokens.extend(block_tokens)
                current_text.append(block.text)
                current_blocks.append(block_idx)
        
        # Don't forget the last chunk
        if current_tokens:
            chunks.append(self._make_chunk(
                current_tokens,
                current_text,
                current_blocks,
                chunk_counter,
                doc_name
            ))
        
        return chunks
    
    def _split_block_with_overlap(
        self,
        tokens: List[int],
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Tuple[str, List[int]]]:
        """Split a large block into chunks with overlap"""
        chunks = []
        start = 0
        step = chunk_size - overlap
        
        # Safety check to prevent infinite loop
        if step <= 0:
            raise ValueError(f"Invalid step size: chunk_size ({chunk_size}) - overlap ({overlap}) must be > 0")
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append((chunk_text, chunk_tokens))
            
            if end >= len(tokens):
                break
            
            # Move start forward by (chunk_size - overlap)
            start += step
        
        return chunks
    
    def _make_chunk(
        self,
        tokens: List[int],
        texts: List[str],
        block_indices: List[int],
        chunk_num: int,
        doc_name: str
    ) -> Chunk:
        """Create a Chunk object from accumulated content"""
        chunk_text = " ".join(texts)
        chunk_id = f"{doc_name}__parser{self.parser}__chunk{chunk_num:04d}"
        
        return Chunk(
            id=chunk_id,
            text=chunk_text,
            tokens=tokens,
            token_count=len(tokens),
            source_blocks=block_indices,
            metadata={
                "parser": self.parser,
                "chunk_num": chunk_num
            }
        )


def parse_and_chunk_pdf(
    pdf_path: Path,
    parser: ParserType,
    chunk_size: int,
    overlap: int,
    output_dir: Path,
    tokenizer: str = "cl100k_base"
) -> Path:
    """
    Complete pipeline: Parse PDF and create chunks
    
    Args:
        pdf_path: Path to PDF file
        parser: Which parser to use
        chunk_size: Maximum tokens per chunk
        overlap: Overlap tokens between chunks
        output_dir: Where to save chunks
        tokenizer: tiktoken encoding name
        
    Returns:
        Path to output JSONL file
    """
    # Initialize parser
    pdf_parser = PDFParser(parser=parser, tokenizer=tokenizer)
    
    # Parse PDF into blocks
    print(f"[{parser}] Parsing {pdf_path.name}...")
    blocks = pdf_parser.parse(pdf_path)
    print(f"[{parser}] Extracted {len(blocks)} blocks")
    
    # Create chunks
    doc_name = pdf_path.stem
    print(f"[{parser}] Creating chunks (size={chunk_size}, overlap={overlap})...")
    chunks = pdf_parser.create_chunks(blocks, chunk_size, overlap, doc_name)
    print(f"[{parser}] Created {len(chunks)} chunks")
    
    # Save to JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{doc_name}__parser_{parser}__cs{chunk_size}__ov{overlap}.jsonl"
    
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    
    print(f"[{parser}] Saved chunks to {output_file}")
    return output_file


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse and chunk PDF documents")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--parser",
        type=str,
        choices=["pymupdf", "unstructured", "pdfplumber"],
        default="pymupdf",
        help="Which parser to use"
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap in tokens")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    
    args = parser.parse_args()
    
    parse_and_chunk_pdf(
        pdf_path=args.pdf_path,
        parser=args.parser,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

