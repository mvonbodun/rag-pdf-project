🏆 Mini Project Competition — Unit 2 Week 4: Evaluating RAG 
Tool (Optional)
Purpose
Instructor
structured QA.
Modal / Openai / Openrouter
Run LLM generations.
logfire
Trace generations and chunk sources
LiteLLM
Switch between embedding models
SentenceTransformers
Embedding with MiniLM / Instructor
FAISS / LanceDB / Turbopuffer
Document retrieval
Cohere Reranker
Improve top-k ranking
Neo4j / Memgraph
Store and visualize Bloom-QA graph
Braintrust
Evaluate and compare system performance
judgy
LLM-as-Judge QA accuracy + Bloom label checking
Pydantic
Validate schema (lesson, QA, Bloom node)
ThreadPoolExecutor
Batch embedding
rank_bm25
IR baseline for comparison
Rich
Console outputs and progress display
🎯 Project Goal
Build a Retrieval-Augmented Generation (RAG) system over long PDF lecture notes. Your system should:
- use any.pdf — DATASET: https://www.kaggle.com/datasets/rrr3try/enterprise-rag-markdown
OR
- Parse and chunk PDFs with different combo parameters. Combo parameters are numerous variation of (Chunking_size, Overlap_size)
HOW TO USE THE RIGHT PARAMETERS WINDOW FOR COMBO PARAMETER:
Choosing chunk_size and chunk_overlap in RAG is not arbitrary though many people treat it that way. You can (and should) systematically select chunking parameters based on:
🧠 First Principles of Chunking
To set good chunk_size and chunk_overlap, think about the unit of meaning in your documents and your retrieval goals.
Chunking Goal
Chunk Size (tokens)
Chunk Overlap (tokens)
Why
Preserve semantic units
100–300
20–50
To avoid breaking sentences, definitions, tables mid-way
Enable dense search
256–512
64–128
Enough for embeddings to capture semantic content
Support answer tracing
≤512
moderate (64–128)
Needed for citations, span evaluation
Maximize retrievability
128–256
high overlap (50%)
Improves recall but increases redundancy
Support long reasoning
512–1024
low or adaptive
Rare—used with summarizers or rerankers
📏 Key Constraints
1. Model Context Window
Your retriever + generator (e.g., Gemini, GPT-4, Mistral) has a limit.
Rule of thumb:
max_context ≈ (k * chunk_size)
→ So if you retrieve top-4, chunk size 1024 → 4096 tokens per call.
If your chunk size is too big, you:
Retrieve fewer documents → lower recall
Blow through token budgets fast → costly, less grounding
2. Unit of Meaning in PDFs (We will delve in depth in upcoming units)
PDFs are visually formatted: paragraphs, sections, bullet lists
Don’t just split every 512 tokens. Instead:
Use layout-aware chunking (via tools like unstructured, PyMuPDF, or pdfplumber)
Chunk by heading, paragraph, or table
Then tokenize and truncate/pad
This often outperforms fixed-size chunking, especially for:
contracts
reports
academic papers
user manuals
🧪 How to Choose Parameters Empirically <— This is Our Approach of overall RAG
You can treat chunk_size and overlap as hyperparameters and tune them empirically using RAG-specific metrics:
🔁 Evaluation loop
Create chunks with a given size/overlap
Build vector store / retriever
Generate synthetic questions for chunks (manually or via Gemini)
Evaluate each config using:
Retrieval metrics:
Recall@K
Precision@K
MRR@K
Generation metrics:
Faithfulness (manual or LLM-as-judge)
Answer accuracy
BLEU / ROUGE if gold answers exist
Compare configs (e.g., 128/64 vs 256/32 vs 512/128)
You can automate this in Colab or in tools like LlamaIndex Playground, LangChain Bench, or your own code.
💡 Example Grid Search Setup
chunk_sizes = [128, 256, 512]
overlaps = [32, 64, 128]

for chunk_size in chunk_sizes:
    for overlap in overlaps:
        chunks = chunk_pdf(document, chunk_size, overlap)
        retriever = build_faiss(chunks)
        eval_metrics = evaluate_rag(retriever, questions)
        log_metrics(chunk_size, overlap, eval_metrics)

​
🧠 Analogy
Choosing chunk_size and overlap_size in RAG is like choosing the size of puzzle pieces:
Too big → one piece covers too much, and you lose granularity
Too small → too many meaningless fragments, hard to match
Overlap = how much the pieces share — helps form clearer context but can create redundancy
- Use Different embedding models to embed into vector db. For example if step 1 has 4 combo of parameters and you are using 3 embeddings models, it will have 12 vector db for each type of combo parameters with each type of embeddings.
- Generate synthetic dataset of questions and label from chunks i.e. synthetic questions and chunks ids for each vector db. For example: continuing the previous example, we will have 12 files for synthetic data which has Questions and Gold label as chunk id.  Make sure to generate Questions from Overlap (usually overlap reveals the true weakness of RAG).
- Evaluate performance using Recall, MRR@1,3,5, Precision@1,3,5.
- Pick the best combinations of embedding models, retrieval methods and chunking methods.
- (Optional) Use Cohere reranking or open-source reranking to improve top-k retrieval. (Optional)
- Evaluate performance using Recall, MRR@1,3,5, Precision@1,3,5. with reranking (we will dive more deeper into it in advance RAG) (Optional)
Attach LLM with the retriever using prompt engineering.
Evaluate using Generation Metrics, such as faithfulness, Relevance, and ground truth. (If scores are lower than using, there is something wrong in the 6th point specifically with prompt engineering)
Test the full pipeline.
Additional Info:
    - Use FAISS or turbopuffer for vector DB.
    - Use retrieval over Synthetic Question and match gold label with top-k retreived chunk ids. (Optional) TRY DIFFERENT RETRIEVAL TO CHECK HOW IT CHANGES SCORES (Optional)
    - Compare the performance of different embedding models, retrieval methods and chunking methods.
    - Use evaluation metrics like Recall, MRR, and Precision to compare the performance of different combinations of methods.
    - Pick the best combinations of embedding models, retrieval methods, and chunking methods.
🧠 Additional Bonus Ideas
Add Additional Techniques over RAG.
Add LLM-as-Judge to evaluate the Question generated for synthetic QA.

Use Logfire to trace generations, Logging, tracing, debugging, and inspecting LLM inputs/outputs at runtime.

Add Braintrust for logging and monitoring comprehensive debugging and evaluation.

!!!!The JSON file sometimes shows incorrect scores - the CLI output shows the true performance results. !!! USE Pydantic to make sure the JSON file is correct.

Use Cohere reranking to improve the best combination of techniques for top-k retrieval and compare without reranking.

I am using openai for various embedding, use LiteLLM or download Open Source Embedding Models for RAG from huggingface to switch to between embedding.

Use openrouter, or openai for LLM generations if needed.
​
Show QA density by diversity using QA metrics.
## 🎯 **Better Synthetic QA Generation Approaches:**

### **1. Multi-Chunk Questions**
Instead of 1 question → 1 chunk, create questions that span multiple related chunks:

```python
# Example: Combine 2-3 related chunks
related_chunks = find_semantically_similar_chunks(chunk, chunks, top_k=3)
question = f"Explain the relationship between {topic_A} and {topic_B} based on the content"
relevant_chunk_ids = [chunk.id for chunk in related_chunks]
```

### **2. Question Type Diversity**
Generate different types of questions:

```python
question_types = [
    "factual": "What is X?",
    "comparative": "How does X differ from Y?", 
    "analytical": "Why does X happen?",
    "summarization": "Summarize the key points about X",
    "multi-hop": "What is X and how does it relate to Y?"
]
```

### **3. Hierarchical Questions**
Create questions at different granularity levels:

```python
# Page-level questions (multiple chunks)
# Section-level questions (few chunks)  
# Paragraph-level questions (single chunk)
```

### **4. Real-World Question Patterns**
Use domain-specific question templates:

```python
academic_patterns = [
    "Define {concept} and explain its significance",
    "Compare and contrast {concept_a} with {concept_b}",
    "What are the main components of {system}?",
    "How does {process} work step by step?"
]
```

### **5. LLM-Generated Question Chains**
Use GPT to create question sequences:

```python
prompt = f"""Based on this content, generate 3 questions:
1. A basic factual question
2. A deeper analytical question  
3. A question connecting this to other concepts

Content: {chunk.text}"""
```

implement any of these approaches to improve the synthetic QA generation.
​
Use feedback classification ("thumbs up/down") using Braintrust to have end-to-end pipeline for RAG system