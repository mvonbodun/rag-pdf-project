# Query FAISS Index Tool

A command-line tool for querying FAISS vector indexes with natural language queries.

## Features

- 🔍 **Natural Language Search**: Query your vector indexes using plain English
- 🎨 **Beautiful Output**: Rich, formatted results with color-coded similarity scores
- 💬 **Interactive Mode**: Run multiple queries in a session
- 📚 **Index Selection Menu**: Browse and select from all available indexes
- 🔄 **Switch Indexes**: Change between indexes without restarting
- 🎯 **Single Query Mode**: Quick one-off searches
- 🤖 **Auto-Embedding**: Automatically uses the same embedding model as the index
- 📊 **Score Visualization**: Color-coded scores (green > 0.85, yellow > 0.75, red < 0.75)

## Installation

The tool uses the same dependencies as the main pipeline. If you can build indexes, you can query them!

Optional: For loading `.env` files automatically:
```bash
pip install python-dotenv
```

## Usage

### Interactive Mode with Index Selection

Start with no index to see a menu of all available indexes:

```bash
python src/query_index.py
```

You'll see a beautiful menu grouped by document:
```
╭─────── 📚 Available Indexes ────────╮
│ Select a FAISS Index                │
│ Found 37 indexes across 2 documents │
╰─────────────────────────────────────╯

ASICS-AW23-Run-Catalog
  [ 1] pdfplumber | cs128_ov32   | ada-002
  [ 2] pdfplumber | cs128_ov32   | openai-large
  ...

Select index (1-37) or 'q' to quit:
```

### Interactive Mode with Specific Index

Start with a specific index:

```bash
python src/query_index.py --index fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small
```

Then type queries at the prompt:
```
Query (k=5): What is the total budget?
Query (k=5): switch              # Switch to a different index!
Query (k=5): k=10               # Adjust result count
Query (k=10): Tell me about immigration appeals
Query (k=10): help              # Show available commands
Query (k=10): quit              # Exit
```

Commands in interactive mode:
- `switch`, `change`, `index` - Change to a different index (shows menu)
- `k=N` - Change the number of results returned
- `help`, `?` - Show help message with all commands
- `quit`, `exit`, `q` - Exit the tool

### Single Query Mode

Run a single query and exit:

```bash
# Full path
python src/query_index.py \
    --index indexes/faiss/fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \
    --query "What is the total budget for FY 2010?" \
    --top-k 5

# Short path (assumes indexes/faiss/ prefix)
python src/query_index.py \
    -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \
    -q "What is the total budget for FY 2010?" \
    -k 5
```

## Arguments

```
-i, --index INDEX       Path to FAISS index directory (required)
                        Can be full path or just the index name
                        Examples:
                        - indexes/faiss/myindex__emb_openai-small
                        - myindex__emb_openai-small (auto-adds prefix)

-q, --query QUERY       Query string (optional)
                        If omitted, runs in interactive mode

-k, --top-k K          Number of results to return (default: 5)
```

## Examples

### Query About Specific Topics

```bash
# Budget information
python src/query_index.py \
    -i fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002 \
    -q "What is the total immigration budget?" \
    -k 3

# Technical specifications
python src/query_index.py \
    -i ASICS-AW23-Run-Catalog__parser_pymupdf__cs256__ov64__emb_openai-small \
    -q "What are the shoe features for trail running?" \
    -k 5
```

### Compare Different Configurations

Query the same content across different index configurations:

```bash
# Compare chunk sizes
python src/query_index.py -i fy10syb__parser_pdfplumber__cs128__ov32__emb_openai-small \
    -q "immigration appeals statistics"

python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \
    -q "immigration appeals statistics"

# Compare embedding models
python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small \
    -q "What is the total budget?"

python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002 \
    -q "What is the total budget?"
```

### Interactive Exploration

```bash
python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_openai-small

# Then explore:
Query (k=5): overview of the document
Query (k=5): k=3
Query (k=3): what are the key statistics?
Query (k=3): tell me about appeals
Query (k=3): quit
```

## Output Format

Results display:
- **Score**: Similarity score (0-1, higher is better)
  - 🟢 Green: > 0.85 (excellent match)
  - 🟡 Yellow: 0.75-0.85 (good match)
  - 🔴 Red: < 0.75 (moderate match)
- **Metadata**: Page number, chunk index (if available)
- **Text**: Chunk content (truncated at 500 chars if longer)
- **Chunk ID**: Internal identifier for the chunk

## How It Works

1. **Load Index**: Reads the FAISS index, chunks, and metadata
2. **Extract Model Info**: Determines which embedding model was used
3. **Initialize Embedder**: Sets up the same embedding model
4. **Embed Query**: Converts your text query to a vector
5. **Search**: Finds the K nearest neighbors in the vector space
6. **Display**: Shows results with formatted, readable output

## Tips

1. **Choose the Right Index**: Use the best-performing configuration from your evaluation
   - Check `RESULTS_SUMMARY.md` for recommendations
   - For fy10syb.pdf: Use `fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002`

2. **Adjust K**: Start with k=5, increase for broader exploration

3. **Refine Queries**: If results aren't relevant, try:
   - More specific questions
   - Different phrasings
   - Including key terms from your document

4. **Check Scores**: 
   - Scores > 0.8 usually indicate strong relevance
   - Scores < 0.6 may indicate query-document mismatch

5. **Interactive Mode**: Use for exploration and iterative refinement

## Troubleshooting

### "OPENAI_API_KEY not found"

Set your OpenAI API key (for OpenAI embedding models):

```bash
export OPENAI_API_KEY='sk-...'
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

### "Index directory not found"

Make sure the index exists:
```bash
ls indexes/faiss/
```

If using a short name, ensure it's in `indexes/faiss/`:
```bash
# These should be equivalent:
python src/query_index.py -i indexes/faiss/my_index__emb_openai-small -q "test"
python src/query_index.py -i my_index__emb_openai-small -q "test"
```

### "sentence-transformers not installed"

If querying an index built with SentenceTransformers:
```bash
pip install sentence-transformers
```

## Integration with Pipeline

This tool is perfect for:
1. **Testing**: Verify your indexes work before running full evaluations
2. **Debugging**: Understand what your indexes are retrieving
3. **Exploration**: Discover what's in your documents
4. **Validation**: Check if the right chunks are being matched
5. **Demonstration**: Show stakeholders how the system works

Use it alongside the evaluation pipeline to:
- Spot-check top-performing configurations
- Understand why certain configs perform better
- Validate that chunking strategies work as expected
