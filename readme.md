# Production-Level RAG, From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pranati-56/Production_RAG/blob/main/Copy_of_production_rag_from_scratch.ipynb)

A from-scratch implementation of an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built to follow a two-day "Production-Level RAG" workshop. 

Unlike typical tutorials that rely on heavy framework abstractions like LangChain or LlamaIndex for core logic, every stage of this pipeline — document parsing, chunking algorithms, vector embeddings, similarity retrieval, augmented prompt generation, and RAGAS evaluation — is implemented **by hand with pure Python, PyTorch/NumPy, sentence-transformers, PyMuPDF, and the Gemini API (`google-genai`)**.

This hands-on design makes all architectural trade-offs visible, measurable, and explicitly tunable.

---

## Table of Contents

- [Overview \& Design Philosophy](#overview--design-philosophy)
- [Grounding Document](#grounding-document)
- [Pipeline Architecture](#pipeline-architecture)
- [1. Setup \& Dependencies](#1-setup--dependencies)
- [2. Document Ingestion \& Data Processing (EDA)](#2-document-ingestion--data-processing-eda)
- [3. Chunking Strategies Explored from Scratch](#3-chunking-strategies-explored-from-scratch)
  - [4.1 Fixed-Size Chunking](#41-fixed-size-chunking)
  - [4.2 Semantic Chunking](#42-semantic-chunking)
  - [4.3 Recursive Chunking](#43-recursive-chunking)
  - [4.4 Structural Chunking](#44-structural-chunking)
  - [4.5 LLM-Based Chunking](#45-llm-based-chunking)
- [4. Empirical Chunking Strategy Comparison](#4-empirical-chunking-strategy-comparison)
- [5. Unit Testing for Chunking Functions](#5-unit-testing-for-chunking-functions)
- [6. Production Chunk Set Construction](#6-production-chunk-set-construction)
- [7. Embedding Generation \& Vector Storage](#7-embedding-generation--vector-storage)
- [8. Vector Retrieval](#8-vector-retrieval)
- [9. Context Augmentation \& Generation](#9-context-augmentation--generation)
- [10. Pipeline Evaluation with RAGAS](#10-pipeline-evaluation-with-ragas)
- [11. Hyperparameter Search](#11-hyperparameter-search)
- [12. Engineer's Choice Decision Matrix](#12-engineers-choice-decision-matrix)
- [13. Production Deployment Architecture](#13-production-deployment-architecture)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)

---

## Overview & Design Philosophy

Production-grade RAG systems require fine-grained control over document chunking, embedding model bounds, retrieval mechanics, and LLM grounding. Framework abstractions often mask critical performance bottlenecks, token truncation edge cases, and unexpected costs.

### Core Principles of This Repository

1. **Zero Framework Bloat for Core Pipeline**: No high-level wrapper libraries (e.g. LangChain, LlamaIndex) in the retrieval and generation path. All data structures, chunkers, tensor operations, and API calls are standard Python functions.
2. **Unified LLM Provider**: Uses the official **Gemini API** (`google-genai` / `gemini-3.1-flash-lite` or `gemini-3.6-flash`) for all LLM calls — LLM-based chunking (§4.5), response generation (§10), and the RAGAS evaluation judge (§12) — managed under **one API key**.
3. **Fully Local, Ungated Embeddings**: Uses open-source `sentence-transformers` models (`all-MiniLM-L6-v2` for semantic chunking, `all-mpnet-base-v2` for production vector index) running locally on PyTorch (CPU or GPU). No gated model licenses or Hugging Face access tokens required.
4. **Deterministic Unit Testing**: Every chunking strategy and formatter function is verified with a custom unit testing harness (`run_tests`) to prevent silent failure modes (e.g., character budget overruns, 0-chunk returns on failed regex matching).
5. **Empirical Evaluation**: Pipeline quality is validated statistically across chunking strategies and evaluated end-to-end using standard RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall, Context Entity Recall).

---

## Grounding Document

The working dataset used throughout the notebook is:
* **Book**: *Human Nutrition: 2020 Edition* (University of Hawai'i at Mānoa, CC BY 4.0)
* **Format**: Digitally-born PDF (~1,200 pages)
* **Characteristics**: Clean digital text without scanning artifacts. Raw PyMuPDF page index 42 maps to the book's printed Page 1 (Chapter 1 start), establishing a baseline for page-offset mapping.

---

## Pipeline Architecture

```
                                  +-----------------------+
                                  |  Human Nutrition PDF  |
                                  +-----------+-----------+
                                              |
                                              v [PyMuPDF (fitz)]
                                  +-----------+-----------+
                                  | Text & Page Extraction|
                                  +-----------+-----------+
                                              |
                                              v [text_formatter & NLTK]
                                  +-----------+-----------+
                                  | Token & Sentence Stats|
                                  +-----------+-----------+
                                              |
        +-------------------------------------+-------------------------------------+
        |                                     |                                     |
        v                                     v                                     v
+---------------+                     +---------------+                     +---------------+
| 5 Chunkers    |                     | Unit Test     |                     | Production    |
| Benchmarked   |                     | Harness       |                     | Sentence      |
| (§4.1 - §4.5) |                     | (§6)          |                     | Chunks (§7)   |
+---------------+                     +---------------+                     +---------------+
                                                                                    |
                                                                                    v [all-mpnet-base-v2]
                                                                            +---------------+
                                                                            | 768-dim Vector|
                                                                            | Embeddings    |
                                                                            +-------+-------+
                                                                                    |
                                                                                    v
                                  +-----------------------+                 +---------------+
                                  | User Query            | --------------> | Dot-Score Top-k|
                                  +-----------------------+                 | Retrieval     |
                                                                            +-------+-------+
                                                                                    |
                                                                                    v
                                  +-----------------------+                 +---------------+
                                  | Augmented Prompt      | <-------------- | Context       |
                                  | Construction          |                 | Assembly      |
                                  +-----------+-----------+                 +---------------+
                                              |
                                              v [Gemini API (google-genai)]
                                  +-----------+-----------+
                                  | Grounded Response     |
                                  +-----------+-----------+
                                              |
                                              v [RAGAS Evaluation (§12)]
                                  +-----------+-----------+
                                  | Faithfulness, Relevancy|
                                  | Precision & Recall    |
                                  +-----------------------+
```

---

## 1. Setup & Dependencies

The project relies on a minimal set of open-source and official provider SDKs:

```bash
pip install -q PyMuPDF sentence-transformers nltk pandas numpy matplotlib tqdm google-genai ragas langchain-google-genai
```

### Dependency Notes
* **PyMuPDF (`fitz`)**: For fast, high-fidelity PDF text and metadata extraction (~10-15x faster than Docling on digital PDFs).
* **sentence-transformers**: Provides local models `all-MiniLM-L6-v2` (384-dim) and `all-mpnet-base-v2` (768-dim).
* **google-genai**: Official Google GenAI SDK for interacting with Gemini models.
* **ragas & langchain-google-genai**: Framework for computing RAG evaluation metrics using Gemini as the judge LLM.

---

## 2. Document Ingestion & Data Processing (EDA)

Document ingestion extracts text on a per-page basis and computes token statistics to prevent silent context truncation during embedding.

### Text Extraction & Cleaning
`open_and_read_pdf` reads PDF pages, computes token count, word count, character count, and sentence count. Whitespace and non-printable line breaks are normalized using `text_formatter`:

```python
def text_formatter(text: str) -> str:
    """Clean and normalize whitespace within a block of text."""
    cleaned_text = text.replace("\n", " ").strip()
    return " ".join(cleaned_text.split())
```

### Key Statistical Insights (Exploratory Data Analysis)
* **Mean page length**: ~287 tokens (~10 sentences).
* **Embedding Model Constraint**: Sentence-transformer models (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`) silently **truncate any input beyond 384 word pieces**.
* **Ingestion Finding**: Since mean page token length (~287 tokens) sits comfortably under 384, page-level or 10-sentence chunking safely avoids vector truncation.

### Parsing Tool Comparison Matrix

| Tool | Recommended Use Case | Trade-offs |
|---|---|---|
| **PyMuPDF (`fitz`)** | Digitally-born text PDFs (no scans) | **Fastest by far** (~10-15x faster than Docling). Cannot read images/scans. |
| **Tesseract (OCR)** | Scanned PDFs, photographs, handwritten documents | Slower; required when text is rendered as pixels. |
| **Docling** | Complex PDFs with tables, formulas, multi-column layouts | ~50x slower than PyMuPDF; preserves rich structural layout and table schemas. |

---

## 3. Chunking Strategies Explored from Scratch

Five distinct chunking algorithms were implemented from first principles to explore their trade-offs in structural integrity, semantic coherence, and compute requirements.

### 4.1 Fixed-Size Chunking
Cuts text every $N$ characters (`chunk_size=500`), splitting on space characters to prevent mid-word fragmentation.

* **Function**: `chunk_fixed_size(text: str, chunk_size: int = 500) -> list[str]`
* **Pros**: Extremely fast, predictable chunk length.
* **Cons**: Blind to sentence boundaries and semantic topic shifts.

### 4.2 Semantic Chunking
Groups sentences sequentially into a chunk as long as each new sentence maintains a cosine similarity above a threshold (`threshold=0.75`) relative to the chunk's **anchor sentence** (the first sentence of that chunk).

* **Function**: `chunk_semantic_page(text: str, model: SentenceTransformer, threshold: float = 0.75) -> list[str]`
* **Pros**: Maintains strict semantic cohesion; prevents topic drifting within a chunk.
* **Cons**: Requires running an embedding model pass per sentence. On dense textbooks, frequent topic changes cause over-segmentation into very small chunks.

### 4.3 Recursive Chunking
Tries to split on coarse separators first (`\n\n` paragraph breaks); if a resulting piece exceeds `max_chars` (default 1000), it recursively splits that piece using finer separators (`\n`, `. `).

* **Function**: `chunk_recursive(text: str, max_chars: int = 1000, separators=("\n\n", "\n", ". ")) -> list[str]`
* **Pros**: Respects document structural hierarchy while establishing a strict upper size ceiling.
* **Cons**: Requires uncollapsed raw text to access original line break structure.

### 4.4 Structural Chunking
Splits the document at explicit section/chapter header markers matching a regular expression pattern (e.g. running textbook chapter headers).

* **Function**: `chunk_structural(pages: list[dict], marker: re.Pattern) -> list[dict]`
* **Pros**: Preserves complete domain-specific sections (e.g., legal clauses, medical records, textbook chapters).
* **Cons**: Chunks have extreme size variance; long chapters can exceed LLM input budgets or cause retrieval latency spikes.

### 4.5 LLM-Based Chunking
Hands splitting decisions directly to **Gemini API** (`gemini-3.1-flash-lite` / `gemini-3.6-flash`). The LLM reads raw text and inserts a unique delimiter tag (`<<<SPLIT>>>`) wherever topic transitions occur.

* **Functions**:
  * `build_llm_chunk_prompt(text: str, delimiter: str) -> str`
  * `split_on_llm_delimiter(marked_text: str, delimiter: str) -> list[str]`
  * `chunk_llm_based(text: str, model: str) -> list[str]`
* **Pros**: Highly intelligent, context-aware topic boundary detection.
* **Cons**: High latency and API cost (~2 orders of magnitude costlier); non-deterministic output.

---

## 4. Empirical Chunking Strategy Comparison

| Strategy | Unit Cut On | Structure Aware? | Semantics Aware? | Relative Cost | Benchmark Signature on 1,200-Page Textbook |
|---|---|---|---|---|---|
| **Fixed-size** | $N$ characters | No | No | Lowest | Most chunks; tight uniform size; zero variance. Fast, blind baseline. |
| **Semantic** | Sentence similarity vs anchor | No | Yes | Medium | Most fragmented chunks; small size; over-splits dense textbook paragraphs. |
| **Structural** | Header regex patterns | Yes | No | Low (regex) | Fewest chunks; highest average length and extreme size variance. |
| **Recursive** | Paragraph $\rightarrow$ Line $\rightarrow$ Sentence | Yes (size capped) | No | Low | Safe middle ground across size, count, and structural alignment. |
| **LLM-Based** | Topic shift via Gemini LLM | Data-driven | Yes | Highest | High quality splits; bounded sample run recommended due to API cost. |

---

## 5. Unit Testing for Chunking Functions

To ensure reliability, a custom unit test runner (`run_tests`) executes 18 deterministic synthetic test cases without requiring external test frameworks like `pytest`.

### Test Suite Coverage
* `test_text_formatter_*`: Verifies whitespace collapsing, empty string handling, and trailing character stripping.
* `test_fixed_size_*`: Ensures zero-length text handling, single-chunk short strings, character limit compliance, and word-boundary safety.
* `test_semantic_chunking_*`: Tests single sentences, paraphrase grouping, and domain-shift splitting using real model embeddings.
* `test_recursive_*`: Verifies paragraph splitting fallback hierarchy and oversized separator-less piece retention.
* `test_structural_*`: Tests regex marker splits and verifies fallback behavior when no marker is matched.
* `test_llm_delimiter_split_*`: Verifies delimiter parsing, missing delimiter retention, and empty slice filtering.

> [!IMPORTANT]
> **Real Bug Resolved by Unit Tests**: An early version of `chunk_structural` silently returned zero chunks when a regex marker failed to match. The test `test_structural_no_marker_present_is_one_chunk` caught this behavior, ensuring non-matching documents safely return the full text as a single chunk.

---

## 6. Production Chunk Set Construction

For the production vector index, the repository builds sentence-grouped chunks:
1. NLTK `punkt` sentence tokenizer splits pages into individual sentences.
2. `split_into_groups` aggregates sentences into fixed blocks of 10 sentences (`NUM_SENTENCE_CHUNK_SIZE = 10`).
3. Chunks with fewer than 30 tokens (`MIN_TOKEN_LENGTH = 30`) — such as standalone header fragments or footnotes — are filtered out to remove low-signal noise.

```python
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30

# Groups sentences into chunks and filters short noise fragments
pages_and_chunks = []
for page in pages_and_texts:
    sentence_groups = split_into_groups(page["sentences"], NUM_SENTENCE_CHUNK_SIZE)
    for group in sentence_groups:
        chunk_text = " ".join(group)
        chunk_tokens = len(chunk_text.split())
        if chunk_tokens >= MIN_TOKEN_LENGTH:
            pages_and_chunks.append({
                "page_number": page["page_number"],
                "sentence_chunk": chunk_text,
                "chunk_token_count": chunk_tokens
            })
```

---

## 7. Embedding Generation & Vector Storage

### Embedding Model Choice
The production index uses **`all-mpnet-base-v2`** from `sentence-transformers`:
* **Output Dimension**: 768 dimensions
* **Execution**: Local PyTorch tensor compute (CPU, GPU, or batched GPU execution)
* **Output Artifact**: Serialized to `text_chunks_and_embeddings_df.csv` containing chunk text, metadata, and 768-dim floating point vector strings.

### Performance Benchmarking
* **CPU Batching**: Functional for small datasets, but slower scaling.
* **GPU Batched (`device="cuda"`, `batch_size=32`)**: Achieves up to ~15-20x speedup over CPU during full document vector encoding.

### Vector Storage Trade-Off Matrix

| Storage Option | Ideal Dataset Scale | Characteristics |
|---|---|---|
| **In-Memory Tensor / NumPy** | $< 100,000$ vectors | **Used in this repository**. Brute-force dot product takes $< 5\text{ ms}$. Zero infrastructure overhead. |
| **FAISS** | $100,000 - 10,000,000$ vectors | Local in-process library supporting IVFFlat and HNSW indexing for fast approximate nearest neighbor search. |
| **Managed Vector DBs** (Pinecone, Qdrant) | $> 10,000,000$ vectors | Fully managed distributed indexing, metadata filtering, and scale management. |
| **Postgres + pgvector** | Any scale (Supabase / Enterprise) | Keeps embeddings and relational metadata side-by-side in PostgreSQL, queryable with standard SQL. Recommended for production web apps. |

---

## 8. Vector Retrieval

Given a user query string:
1. The query is embedded using `all-mpnet-base-v2` into a 768-dim tensor.
2. Vector similarity is calculated against all stored chunk embeddings using PyTorch dot-product matrix multiplication (`util.dot_score`).
3. Top-$k$ chunks ($k=5$) with highest dot-scores are retrieved alongside their page number citations.

```python
def retrieve_relevant_resources(
    query: str,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict],
    n_resources_to_return: int = 5
) -> tuple[torch.Tensor, list[int]]:
    """Embed query and return top-k dot-score matching chunk indices and scores."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices
```

---

## 9. Context Augmentation & Generation

The `prompt_formatter` constructs a structured prompt injecting retrieved context chunks, page metadata, and instructions before submitting to the Gemini API (`gemini-3.1-flash-lite` / `gemini-3.6-flash`).

### Prompt Format
```
Based on the following context items, please answer the query.
Give a clear, detailed response grounded ONLY in the context provided.

Context items:
- [Page 5]: Macronutrients are nutrients needed in large amounts...
- [Page 6]: Carbohydrates, lipids, and proteins are energy-yielding...

Query: What are the macronutrients?
Answer:
```

### End-to-End Execution (`ask`)
The `ask` function coordinates query embedding, retrieval, prompt formatting, and Gemini API invocation:

```python
def ask(
    query: str,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict],
    temperature: float = 0.2,
    return_context: bool = False
):
    scores, indices = retrieve_relevant_resources(query, embeddings, pages_and_chunks)
    context_items = [pages_and_chunks[i] for i in indices]
    prompt = prompt_formatter(query=query, context_items=context_items)
    response = generate_text(prompt=prompt, temperature=temperature)
    return response, context_items
```

---

## 10. Pipeline Evaluation with RAGAS

RAG pipeline quality is quantitatively assessed using **RAGAS (Retrieval-Augmented Generation Assessment System)**.

### Gemini as the RAGAS Judge
RAGAS defaults to OpenAI as a judge model. To maintain a **single API provider philosophy**, the evaluation setup configures a custom Gemini-backed judge via `langchain-google-genai` (`ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings`).

### Metrics Evaluated

| Metric | Focus | Description |
|---|---|---|
| **Context Precision** | Retrieval | Ratio of relevant chunks among all retrieved top-$k$ items. |
| **Context Recall** | Retrieval | Measure of whether all ground-truth facts were retrieved. |
| **Faithfulness** | Generation | Verification that the answer contains ONLY facts present in retrieved context (hallucination check). |
| **Answer Relevancy** | Generation | Quantifies how directly the generated answer addresses the user query. |
| **Context Entity Recall** | Retrieval | Percentage of ground-truth named entities captured in retrieved context. |

```python
# Custom Gemini JSON evaluation wrapper for standalone metric calculations
def faithfulness_score(answer: str, contexts: list[str]) -> float: ...
def answer_relevancy_score(question: str, answer: str) -> float: ...
def context_precision_score(question: str, contexts: list[str]) -> float: ...
def context_recall_score(ground_truth: str, contexts: list[str]) -> float: ...
```

---

## 11. Hyperparameter Search

The repository provides an automated grid-search harness (`build_index` + `ask_with_model`) to evaluate combinations of chunking strategies, chunk sizes, embedding models (`all-mpnet-base-v2` vs. `all-MiniLM-L6-v2`), and top-$k$ retrieval counts against a hand-curated ground-truth spreadsheet.

---

## 12. Engineer's Choice Decision Matrix

| Stage | Available Options | Recommended Selection Strategy |
|---|---|---|
| **Document Parsing** | PyMuPDF / Tesseract / Docling | **PyMuPDF** for clean digital text; **Tesseract** for scans/OCR; **Docling** for complex embedded tables. |
| **Chunking Strategy** | Fixed / Semantic / Structural / Recursive / LLM-Based | **Recursive** as default baseline; **Structural** for legal/structured forms; **LLM-based** for small context-shifting chats. |
| **Embedding Model** | MiniLM / mpnet / OpenAI / Qwen / Gemini | **`all-mpnet-base-v2`** for local open-source quality; commercial API embeddings for zero GPU management. |
| **Vector Storage** | In-Memory Tensor / FAISS / Qdrant / Supabase pgvector | **Tensor** for $< 100\text{k}$ vectors; **pgvector (Postgres)** for relational web production apps. |
| **LLM Generation** | Gemini API / Quantized Local Gemma | **Gemini API** for high quality and speed; **Local Gemma** when data privacy requires local execution. |
| **Evaluation** | Chunk stats vs End-to-End RAGAS | **Chunk stats** for fast pre-filtering; **RAGAS + ground truth set** for final production deployment signal. |

---

## 13. Production Deployment Architecture

Transitioning from this Colab prototype to a production web application involves four decoupled stack layers:

1. **Database & Vector Store (Supabase / PostgreSQL + `pgvector`)**:
   - `chunks` table storing chunk text, document metadata, and `embedding vector(768)`.
   - SQL match function (`match_documents`) executing cosine distance / IVFFlat vector search in PostgreSQL.
2. **Ingestion Service (`ingest.py`)**:
   - Standalone CLI/worker script running PyMuPDF parsing, sentence chunking, embedding generation, and bulk upserting to Supabase.
3. **Backend API Route (FastAPI / Next.js `route.ts`)**:
   - Accepts user chat query, embeds query vector via `sentence-transformers`, executes `match_documents` RPC call, builds prompt, calls Gemini API, and streams response with page citations.
4. **Frontend UI (Streamlit / Lovable / React)**:
   - Modern chat interface displaying streaming responses, original document page references, and similarity score metadata.

---

## Getting Started

### Prerequisites
* Python 3.10 or higher
* Recommended: NVIDIA GPU with CUDA support for accelerated local embeddings
* Gemini API Key ([Get an API key from Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pranati-56/Production_RAG.git
   cd Production_RAG
   ```

2. **Install dependencies**:
   ```bash
   pip install PyMuPDF sentence-transformers nltk pandas numpy matplotlib tqdm google-genai ragas langchain-google-genai
   ```

3. **Set your Gemini API Key**:
   * **Linux/macOS**: `export GEMINI_API_KEY="your-api-key-here"`
   * **Windows (PowerShell)**: `$env:GEMINI_API_KEY="your-api-key-here"`

4. **Launch the Notebook**:
   Open `Copy_of_production_rag_from_scratch.ipynb` in Google Colab or VS Code Jupyter Extension.

---

## Repository Structure

```
.
├── Copy_of_production_rag_from_scratch.ipynb  # Primary from-scratch RAG notebook & benchmark suite
├── README.md                                  # Production RAG architecture & documentation
└── text_chunks_and_embeddings_df.csv          # (Generated) Production chunks & 768-dim embeddings cache
```

---

## License

The code in this repository is open for educational and production adaptation. The grounding textbook (*Human Nutrition: 2020 Edition*) is licensed under **CC BY 4.0**.
