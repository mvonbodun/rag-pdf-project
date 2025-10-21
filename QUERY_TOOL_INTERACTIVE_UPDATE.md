# Query Index Tool - Interactive Mode Update

## 🎉 New Features Added

The query tool now has enhanced interactive mode with:

### ✨ **1. Index Selection Menu**
When you run the tool without specifying an index, it shows a beautiful menu of all available indexes grouped by document:

```bash
python src/query_index.py
```

You'll see:
```
╭─────── 📚 Available Indexes ────────╮
│ Select a FAISS Index                │
│ Found 37 indexes across 2 documents │
╰─────────────────────────────────────╯

ASICS-AW23-Run-Catalog
  [ 1] pdfplumber | cs128_ov32   | ada-002
  [ 2] pdfplumber | cs128_ov32   | openai-large
  ...

fy10syb
  [19] pdfplumber | cs128_ov32   | ada-002
  ...

Select index (1-37) or 'q' to quit:
```

### ✨ **2. Switch Index Command**
While querying, you can switch to a different index without restarting:

```
Query (k=5): switch
```

This shows the index selection menu again, and once you pick a new index, it:
- Loads the new index
- Initializes the correct embedding model
- Continues querying with the new index

### ✨ **3. Help Command**
New help command in interactive mode:

```
Query (k=5): help
```

Shows all available commands:
- `switch` - Change to a different index
- `k=N` - Set number of results
- `quit` - Exit
- `help` - Show help

## Usage Examples

### Start with No Index (Shows Menu)

```bash
python src/query_index.py
```

1. Pick index from menu (e.g., 25 for fy10syb best config)
2. Enter queries
3. Type `switch` to change documents
4. Pick new index
5. Continue querying

### Start with Specific Index

```bash
python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002
```

- Skips the menu
- Starts querying immediately
- Can still use `switch` command

### Single Query Mode (Unchanged)

```bash
python src/query_index.py \
    -i fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002 \
    -q "What is the budget?" \
    -k 5
```

## Interactive Session Example

```
$ python src/query_index.py

[Index menu appears]
Select index (1-37): 25
✓ Selected: fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002
✓ Loaded index... (250 vectors)

Query (k=5): What is the total budget for FY 2010?
[Shows results from fy10syb]

Query (k=5): switch

[Index menu appears again]
Select index (1-37): 6
✓ Selected: ASICS-AW23-Run-Catalog__parser_pdfplumber__cs256__ov64__emb_openai-small
✓ Loaded index... (178 vectors)
✓ Switched to: ASICS-AW23-Run-Catalog__parser_pdfplumber__cs256__ov64__emb_openai-small

Query (k=5): What are the features of GEL-NIMBUS shoes?
[Shows results from ASICS catalog]

Query (k=5): k=10
✓ Set k=10

Query (k=10): help
[Shows help panel]

Query (k=10): quit
Goodbye!
```

## Benefits

### 1. **Easy Comparison**
Compare how different indexes handle the same query:
- Switch between chunk sizes
- Compare embedding models
- Test parser differences

### 2. **Exploration**
Explore multiple documents without restarting:
- Query fy10syb budget data
- Switch to ASICS catalog
- Query product features
- Switch back

### 3. **Configuration Testing**
Validate evaluation results interactively:
- Test best config
- Compare with alternatives
- Understand why one config performs better

### 4. **User-Friendly**
No need to remember index names:
- Browse available indexes
- See full configuration details
- Pick by number

## Implementation Details

### New Functions

1. **`list_available_indexes()`**
   - Scans `indexes/faiss/` directory
   - Returns sorted list of valid indexes

2. **`select_index_interactive()`**
   - Groups indexes by PDF name
   - Displays formatted menu
   - Handles user selection
   - Validates choices

3. **Updated `interactive_mode()`**
   - Accepts optional `index_dir` parameter
   - Calls selection menu if `None`
   - Handles `switch` command
   - Reloads index and embedder

### Command Processing

Interactive mode now handles:
- `quit`, `exit`, `q` - Exit
- `switch`, `change`, `index` - Change index
- `k=N` - Adjust result count
- `help`, `?` - Show help
- Any other text - Treat as query

### Error Handling

- Validates index selection (1-N)
- Handles keyboard interrupts gracefully
- Falls back to current index if switch fails
- Clear error messages

## Arguments Update

The `--index` argument is now **optional**:

```bash
# Old (still works)
python src/query_index.py -i fy10syb__parser_pdfplumber__cs512__ov128__emb_ada-002

# New (shows menu)
python src/query_index.py
```

## Files Modified

- `src/query_index.py`:
  - Added `list_available_indexes()` function (20 lines)
  - Added `select_index_interactive()` function (70 lines)
  - Updated `interactive_mode()` to accept optional index_dir
  - Added `switch` command handling
  - Added `help` command
  - Updated argument parser to make `--index` optional
  - Enhanced help text

## Backward Compatibility

✅ All existing usage patterns still work:
- Single query with specified index
- Interactive with specified index
- Short index names
- Full index paths

➕ New usage patterns:
- Interactive with menu selection
- Switch between indexes
- Help command

## Testing

Successfully tested:
- ✅ Menu display with 37 indexes
- ✅ Index selection (1-37)
- ✅ Grouped display (ASICS + fy10syb)
- ✅ Formatted output (parser | chunks | embedding)
- ✅ Load and query functionality
- ✅ Backward compatibility

Next: Test the `switch` command in a live session!
