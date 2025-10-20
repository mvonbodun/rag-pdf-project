# Step 4: Synthetic QA Generation - COMPLETE ✓

**Date:** October 20, 2025  
**Status:** Successfully generated 72 synthetic QA pairs across 6 chunk configurations

---

## Overview

Step 4 implemented synthetic question-answer pair generation to create evaluation datasets for testing retrieval systems. We used OpenAI's `gpt-4o-mini` model to generate diverse question types that stress-test different aspects of our RAG pipeline.

## Implementation Details

### Module: `src/gen_synth_qa.py`

**Key Features:**
- 4 question types targeting different retrieval challenges
- Batch processing for all chunk configurations
- Rich terminal UI with progress tracking
- Structured output with gold labels (relevant_chunk_ids)

### Question Types Generated

1. **Factual Questions (5 per config)**
   - Purpose: Test basic retrieval
   - Example: "Who prepared the Statistical Year Book for FY 2010?"
   - Gold label: Single chunk ID
   - Difficulty: Easy-Medium

2. **Analytical Questions (3 per config)**
   - Purpose: Test understanding and reasoning
   - Example: "What implications can be drawn about the trends in immigration appeal decisions?"
   - Gold label: Single chunk ID (requires deeper comprehension)
   - Difficulty: Medium-Hard

3. **Multi-hop Questions (2 per config)**
   - Purpose: Test complex retrieval across multiple chunks
   - Example: "How do the failure to appear rates relate to asylum case completion rates?"
   - Gold label: 3 consecutive chunk IDs
   - Difficulty: Hard

4. **Boundary Questions (2 per config)**
   - Purpose: Test chunk overlap handling
   - Example: "What does the disclaimer indicate and how is information structured?"
   - Gold label: 2 adjacent chunk IDs
   - Difficulty: Medium-Hard

## Results Summary

### QA Files Generated

| Chunk Configuration | Chunks | Factual | Analytical | Multi-hop | Boundary | Total |
|---------------------|--------|---------|------------|-----------|----------|-------|
| fy10syb__parser_pdfplumber__cs128__ov32  | 850 | 5 | 3 | 2 | 2 | **12** |
| fy10syb__parser_pdfplumber__cs256__ov64  | 433 | 5 | 3 | 2 | 2 | **12** |
| fy10syb__parser_pdfplumber__cs512__ov128 | 250 | 5 | 3 | 2 | 2 | **12** |
| fy10syb__parser_pymupdf__cs128__ov32     | 689 | 5 | 3 | 2 | 2 | **12** |
| fy10syb__parser_pymupdf__cs256__ov64     | 343 | 5 | 3 | 2 | 2 | **12** |
| fy10syb__parser_pymupdf__cs512__ov128    | 199 | 5 | 3 | 2 | 2 | **12** |
| **TOTAL** | **2,764** | **30** | **18** | **12** | **12** | **72** |

### Output Structure

Each QA file contains JSONL formatted items with:
```json
{
  "question": "The question text",
  "answer": "The expected answer",
  "relevant_chunk_ids": [0, 1],  // Gold labels for evaluation
  "question_type": "factual|analytical|multi-hop|boundary",
  "difficulty": "easy|medium|hard",
  "chunk_config": "fy10syb__parser_pymupdf__cs256__ov64",
  "metadata": {
    "source_chunk_id": 0  // or "source_chunk_ids": [0, 1, 2]
  }
}
```

### Files Created

**QA Files** (6 total, 12 questions each):
- `data/qa/fy10syb__parser_pdfplumber__cs128__ov32__qa.jsonl`
- `data/qa/fy10syb__parser_pdfplumber__cs256__ov64__qa.jsonl`
- `data/qa/fy10syb__parser_pdfplumber__cs512__ov128__qa.jsonl`
- `data/qa/fy10syb__parser_pymupdf__cs128__ov32__qa.jsonl`
- `data/qa/fy10syb__parser_pymupdf__cs256__ov64__qa.jsonl`
- `data/qa/fy10syb__parser_pymupdf__cs512__ov128__qa.jsonl`

**Total Size:** ~40 KB of QA data

## Performance Metrics

- **Model Used:** OpenAI gpt-4o-mini
- **Total Generation Time:** ~5 minutes for all 6 configurations
- **Average Time per Config:** ~50 seconds
- **Average Time per Question:** ~4 seconds
- **Success Rate:** 100% (72/72 questions generated successfully)

## Technical Details

### LLM Configuration

```yaml
# From configs/providers.yaml
llms:
  cost:
    provider: openai
    model: gpt-4o-mini  # Used for QA generation
  quality:
    provider: openai
    model: gpt-4o        # Available for higher quality if needed
```

### Generation Strategy

1. **Factual & Analytical:** Sample from different parts of the document to ensure diversity
2. **Multi-hop:** Select 2-3 consecutive chunks to force cross-chunk reasoning
3. **Boundary:** Use adjacent chunks to test overlap handling
4. **Temperature:** 0.7 for creative but focused question generation
5. **Max Tokens:** 1000 per response to allow detailed answers

### Error Handling

- JSON extraction from LLM responses (handles markdown code blocks)
- Fallback mechanisms for failed generations
- Graceful skipping of problematic chunks
- Clear error reporting in batch mode

## Example Questions

### Factual (Easy)
**Q:** "Who prepared the Statistical Year Book for the U.S. Department of Justice in FY 2010?"  
**A:** "The Office of Planning, Analysis, & Technology."  
**Chunks:** [0]

### Analytical (Medium)
**Q:** "What might the trend in the percentage of represented cases suggest about accessibility of legal representation?"  
**A:** [Requires reasoning about trends and implications]  
**Chunks:** [171]

### Multi-hop (Hard)
**Q:** "What types of statistical information does the Year Book provide regarding immigration court proceedings, and how does it differentiate between asylum cases and other types of relief?"  
**A:** [Requires synthesizing information from multiple sections]  
**Chunks:** [0, 1, 2]

### Boundary (Medium)
**Q:** "What does the disclaimer indicate about legal authority, and how is this information structured in the Table of Contents?"  
**A:** [Requires connecting information across adjacent chunks]  
**Chunks:** [0, 1]

## CLI Usage

```bash
# Single file mode
python -m src.gen_synth_qa \
  --chunk-file data/processed/fy10syb__parser_pymupdf__cs256__ov64.jsonl \
  --model gpt-4o-mini

# Batch mode (all chunk files)
python -m src.gen_synth_qa \
  --chunk-dir data/processed \
  --output-dir data/qa \
  --model gpt-4o-mini \
  --num-factual 5 \
  --num-analytical 3 \
  --num-multi-hop 2 \
  --num-boundary 2
```

## Next Steps: Step 5 - Retrieval Evaluation

Now that we have synthetic QA pairs with gold labels, we can:

1. **Implement `src/eval_retrieval.py`:**
   - Load QA pairs and their gold labels
   - Query all 18 vector indexes (6 chunk configs × 3 embeddings)
   - Compute retrieval metrics: Recall@K, Precision@K, MRR@K
   - Compare configurations to find best performers

2. **Evaluation Matrix:**
   - 6 chunk configurations × 3 embedding models = 18 systems
   - 72 questions per system = 1,296 total retrievals
   - Metrics at K=1, K=3, K=5, K=10

3. **Expected Insights:**
   - Which chunk size works best for different question types?
   - How does parser choice affect retrieval quality?
   - Which embedding model performs best overall?
   - Do larger chunks help with multi-hop questions?
   - Does overlap improve boundary question retrieval?

---

## Configuration Used

**Parsers:** 2 (pymupdf, pdfplumber)  
**Chunk Sizes:** 3 (128, 256, 512 tokens)  
**Overlap Ratios:** 25% consistent (32, 64, 128 tokens)  
**LLM Model:** gpt-4o-mini from OpenAI  
**Question Types:** 4 (factual, analytical, multi-hop, boundary)  
**Questions per Config:** 12 (5+3+2+2)  

**Total Configurations:** 6 chunk configs  
**Total Questions:** 72 synthetic QA pairs  
**Total QA Files:** 6 JSONL files in `data/qa/`

---

## Key Learnings

1. **Question Diversity Matters:** Different question types stress different aspects of retrieval
2. **Gold Labels Enable Precision:** Tracking `relevant_chunk_ids` allows exact evaluation
3. **Boundary Questions Are Critical:** They specifically test chunk overlap handling
4. **Multi-hop Tests Complexity:** These questions reveal whether the system can retrieve multiple relevant chunks
5. **Automated Generation Scales:** LLM-based generation creates diverse, high-quality questions quickly

## Status: ✅ COMPLETE

Step 4 is complete! All 72 synthetic QA pairs have been generated with proper gold labels. We're ready to proceed to Step 5: Retrieval Evaluation.
