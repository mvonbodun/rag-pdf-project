# Layout-Aware Chunking: Current vs Improved Implementation

## What We Have Now (Basic)

### PyMuPDF
```python
# Current: Basic text splitting
text = page.get_text("text")
paragraphs = text.split("\n\n")  # Simple double-newline split
```

### pdfplumber  
```python
# Current: Tables + text splitting
tables = page.extract_tables()  # Good!
text = page.extract_text()
paragraphs = text.split("\n\n")  # Still basic
```

## What We Should Add (True Layout-Aware)

### Improved PyMuPDF (Use dict mode for structure)
```python
# BETTER: Use dict mode to get blocks with bbox and fonts
blocks = page.get_text("dict")["blocks"]

for block in blocks:
    if block["type"] == 0:  # text block
        # Analyze font sizes to identify headings
        lines = block["lines"]
        for line in lines:
            spans = line["spans"]
            avg_font_size = sum(s["size"] for s in spans) / len(spans)
            is_bold = any("Bold" in s["font"] for s in spans)
            
            if avg_font_size > 14 or is_bold:
                block_type = "heading"
            else:
                block_type = "paragraph"
```

### Improved pdfplumber (Use layout objects)
```python
# BETTER: Use layout analysis
words = page.extract_words()
lines = page.extract_text_lines()  # Preserves line structure

# Group into semantic blocks using y-coordinates
# Detect indentation for lists
# Identify heading by font/size if available
```

### unstructured (Already Good!)
```python
# ALREADY LAYOUT-AWARE with strategy="hi_res"
elements = partition_pdf(
    filename=str(pdf_path),
    strategy="hi_res",  # Uses computer vision for layout
    infer_table_structure=True
)
# Returns: Title, NarrativeText, ListItem, Table, etc.
```

## Do We Need to Improve?

### For Your Learning Project: **YES, but maybe later**

**Current Status:**
- ✅ Basic layout awareness (paragraphs, tables)
- ✅ Working pipeline for comparison
- ❌ Not using advanced layout features

**Recommendation:**
1. **For now**: Continue with current implementation to complete the pipeline
2. **Later** (after Step 5-6): Come back and enhance layout-awareness
3. Compare: Basic layout-aware vs. Enhanced layout-aware

### Why This Matters

The REQUIREMENTS emphasize this because:
- **Breaking mid-sentence** → Poor chunk quality
- **Splitting tables** → Loss of structured data
- **Ignoring headings** → Loss of context hierarchy

### Quick Fix Option

We can add a `layout_mode` parameter:
- `basic` (current): Fast, simple paragraph splitting
- `enhanced`: Use font analysis, bbox, proper heading detection
- `unstructured`: Full computer vision (slowest, most accurate)

Would you like me to:
1. **Enhance PyMuPDF** to use font/bbox analysis for better heading detection?
2. **Enhance pdfplumber** to use layout coordinates for semantic grouping?
3. **Keep current** and note this as future improvement?
4. **Proceed to Step 3** and revisit this after seeing retrieval results?

## My Recommendation

**Proceed with current implementation** because:
- It's already better than naive text splitting
- You'll learn from comparing 2 parsers (pymupdf vs pdfplumber)
- After you see retrieval metrics (Step 5), you can decide if layout-awareness is the bottleneck
- This is the empirical approach the REQUIREMENTS recommend!

Then if pdfplumber performs significantly better, it suggests layout matters, and we enhance PyMuPDF to match.

What do you think?
