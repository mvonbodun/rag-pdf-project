# Query Index Tool - Summary

## What Was Created

A new standalone script `src/query_index.py` that allows you to query any FAISS vector index with natural language queries.

## Features

✅ **Natural Language Search** - Query indexes using plain English
✅ **Two Modes:**
   - **Interactive**: Run multiple queries in a session
   - **Single Query**: Quick one-off searches
✅ **Auto-Detection** - Automatically uses the same embedding model as the index
✅ **Beautiful Output** - Rich formatting with color-coded similarity scores:
   - 🟢 Green: Score > 0.85 (excellent match)
   - 🟡 Yellow: Score 0.75-0.85 (good match)
   - 🔴 Red: Score < 0.75 (moderate match)
✅ **Smart Path Handling** - Can use short names (auto-adds `indexes/faiss/` prefix)

## Quick Start

### Single Query

```bash
# Using short index name
python src/query_index.py \
    -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \
    -q "What is the total budget for FY 2010?" \
    -k 5

# Using full path
python src/query_index.py \
    --index indexes/faiss/ASICS-AW23-Run-Catalog__parser_pdfplumber__cs256__ov64__emb_openai-small \
    --query "What are the features of the GEL-NIMBUS shoes?" \
    --top-k 3
```

### Interactive Mode

```bash
python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small
```

Then run queries interactively:
```
Query (k=5): What is the budget?
Query (k=5): k=10
Query (k=10): Tell me about appeals
Query (k=10): quit
```

## Command Reference

```
Arguments:
  -i, --index INDEX       Path to FAISS index directory (required)
  -q, --query QUERY       Query string (optional, triggers single-query mode)
  -k, --top-k K          Number of results (default: 5)
  
Interactive Commands:
  k=N                     Change number of results
  quit, exit, q          Exit interactive mode
```

## Example Output

```
✓ Loaded index: ASICS-AW23-Run-Catalog__parser_pdfplumber__cs256__ov64__emb_openai-small
  Vectors: 178
  Dimensions: 1536
  Chunks: 178
Using OpenAI model: text-embedding-3-small
Embedding query...
Searching index...

╭─────────────────────── 🔍 Search Results ───────────────────────╮
│ Query: What are the features of the GEL-NIMBUS shoes?           │
│ Index: ASICS-AW23-Run-Catalog__parser_pdfplumber__cs256__ov64   │
│ Results: 3                                                      │
╰─────────────────────────────────────────────────────────────────╯

Result 1
Score: 0.7290 • No metadata
╭──────────── ASICS-AW23-Run-Catalog__parserpdfplumber__chunk0054 ────────────╮
│ GEL-NIMBUS® 25 FEATURES                                                      │
│ TheGEL-NIMBUS®25shoe'ssoftcushioningpropertieshelpyoufeellikeyou're         │
│ landingonclouds...                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Use Cases

1. **Testing** - Verify indexes work before full evaluation
2. **Debugging** - Understand what your indexes retrieve
3. **Exploration** - Discover document content
4. **Validation** - Check if chunks match queries correctly
5. **Demonstration** - Show stakeholders how the system works
6. **Comparison** - Query different configurations to compare results

## How It Works

1. Loads the FAISS index, chunks, and metadata
2. Extracts the embedding model info from metadata
3. Initializes the same embedding model
4. Embeds your query text into a vector
5. Searches the index for K nearest neighbors
6. Displays results with scores and chunk content

## Integration with Pipeline

Works seamlessly with the evaluation pipeline:

```bash
# 1. Build indexes for your PDF
python src/run_pipeline.py --pdf data/raw/mydoc.pdf

# 2. Query the best configuration
python src/query_index.py \
    -i mydoc__parser_pdfplumber__cs512__ov128__emb_ada-002 \
    -q "Your question here"
```

## Documentation

- **Full Guide**: `QUERY_TOOL.md` - Comprehensive documentation with examples
- **README**: Updated with quick reference section
- **CLI Help**: `python src/query_index.py --help`

## Requirements

- Same dependencies as the main pipeline
- OpenAI API key (for OpenAI embedding models)
- Optional: `python-dotenv` for automatic `.env` loading

## Testing

Successfully tested with:
- ✅ fy10syb.pdf index (text-embedding-3-small)
- ✅ ASICS-AW23-Run-Catalog.pdf index (text-embedding-3-small)
- ✅ Single query mode
- ✅ Score color coding
- ✅ Auto path resolution (short names)

## Files Created/Modified

1. **Created**: `src/query_index.py` (436 lines) - Main query tool
2. **Created**: `QUERY_TOOL.md` (200+ lines) - Full documentation
3. **Modified**: `README.md` - Added quick reference section
4. **Created**: `QUERY_INDEX_SUMMARY.md` (this file)

## Next Steps

You can now:
1. Query any of your 36 FAISS indexes interactively
2. Compare results across different configurations
3. Validate that your best config (from evaluation) performs well in practice
4. Explore document content through natural language queries
5. Use it as a demonstration tool for stakeholders
