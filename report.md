# Developing an FAQ Chatbot Using LangChain and LLM APIs

**Student:** Ahmad  
**Course:** Artificial Intelligence — Study.com  
**Date:** June 5, 2026  

---

## 1. Name and Purpose of the Chatbot

**RoombaBot** is an FAQ chatbot designed to answer customer support questions about
iRobot Roomba robot vacuums. The purpose of RoombaBot is to give Roomba owners
instant, accurate answers to common questions — covering setup, Wi-Fi connectivity,
cleaning performance, maintenance, troubleshooting, battery life, compatibility, and
warranty — without requiring them to search through a manual or wait for a human
support agent. By grounding every response in a curated FAQ knowledge base rather than
the model's general training data, RoombaBot minimizes hallucination and ensures
answers remain relevant to the specific product. The chatbot runs as a command-line
application entirely on the user's local machine with no internet connection or API
key required after the initial model download, demonstrating how modern open-source
NLP tools can deliver accurate, context-aware customer service at zero ongoing cost.

---

## 2. NLP/LLM Methods Used

RoombaBot uses a **Retrieval-Augmented Generation (RAG)** architecture, which combines
dense vector retrieval with a generative language model (Lewis et al., 2020).

### 2.1 Text Embeddings

Each FAQ entry (question + answer pair) is converted into a dense vector representation
using the `sentence-transformers/all-MiniLM-L6-v2` model, a compact 22 M-parameter
transformer that maps text to a 384-dimensional semantic vector space (Reimers &
Gurevych, 2019). Sentences with similar meaning are positioned close together in this
space regardless of exact wording, allowing the chatbot to match user questions to
relevant FAQs even when the phrasing differs from the dataset.

### 2.2 Similarity Search (FAISS)

The embedded FAQ vectors are indexed in a **FAISS** (Facebook AI Similarity Search)
vector store (Johnson et al., 2019). When a user submits a question, it is embedded
with the same model and FAISS performs a cosine-similarity search to return the top-4
most relevant FAQ chunks. This replaces brittle keyword matching with robust semantic
understanding.

### 2.3 Prompt Engineering and Grounded Generation

The top-4 retrieved FAQ entries are injected into a system prompt that instructs the
LLM to answer using only the provided context. This grounding technique — known as
context-constrained generation — significantly reduces the risk of hallucinated answers.
The model is explicitly told to respond with a support referral if the answer is not
present in the retrieved context, ensuring the chatbot fails gracefully.

### 2.4 Large Language Model (Llama 3.2 via Ollama)

Meta's **Llama 3.2** (3B parameters), served locally through **Ollama**, is used for
response generation. Llama 3.2 is a state-of-the-art open-weight instruction-tuned
model that produces coherent, natural-language answers from the retrieved context
(Meta AI, 2024). Running the model through Ollama means all inference happens on the
user's machine with no API key, no usage fees, and no data leaving the device.
Temperature is set to 0 for consistent, deterministic output appropriate to a
customer-support context.

### Why These Methods Were Chosen

RAG was chosen over fine-tuning because the FAQ dataset is small (54 entries) and
product-specific. Fine-tuning a large model on 54 examples would not meaningfully
improve behavior and would require significant compute. RAG allows the knowledge base
to be updated simply by editing the CSV — no retraining needed. FAISS was selected for
its speed and seamless LangChain integration. Ollama was chosen to eliminate API
dependencies entirely, making the project reproducible on any machine without an
account or payment method.

---

## 3. Dataset Information

### 3.1 Dataset Source

The dataset was curated from iRobot's official customer support documentation,
available at [homesupport.irobot.com](https://homesupport.irobot.com). The FAQ entries
reflect real questions and answers published by iRobot for Roomba robot vacuums
(iRobot, 2025). The curated entries were assembled into a structured CSV file
(`faq_data.csv`) for use in this project.

### 3.2 Number of Records

The dataset contains **54 FAQ entries**.

### 3.3 Number of Features

The dataset has **3 features** (columns).

### 3.4 Description of Features

| Feature Name | Description | Data Type |
|---|---|---|
| `question` | The customer-facing FAQ question | String |
| `answer` | The detailed answer to the question | String |
| `category` | Thematic grouping of the FAQ entry | String (categorical) |

**Category distribution:**

| Category | Count |
|---|---|
| Getting Started | 8 |
| Wi-Fi & App | 8 |
| Cleaning Performance | 8 |
| Maintenance | 8 |
| Troubleshooting | 8 |
| Battery & Charging | 6 |
| Compatibility | 4 |
| Warranty & Support | 4 |
| **Total** | **54** |

### 3.5 Preprocessing Steps

1. **Loading**: The CSV is read into a pandas DataFrame using `pd.read_csv()`.
2. **Document construction**: Each row is converted into a LangChain `Document` object.
   The `question` and `answer` columns are concatenated into a single string in the
   format `"Q: {question}\nA: {answer}"` to preserve the relational context of the
   pair during retrieval.
3. **Metadata tagging**: The `category` column is stored as document metadata, enabling
   category-filtered retrieval in future extensions.
4. **Embedding**: Each document's text is encoded by `all-MiniLM-L6-v2` into a
   384-dimensional float vector. No manual text normalization was applied beforehand, as
   transformer models handle casing and punctuation implicitly through subword
   tokenization.
5. **Indexing**: The vectors are stored in an in-memory FAISS index using cosine
   similarity for retrieval.

---

## 4. Libraries, Toolkits, and Frameworks

| Library | Role |
|---|---|
| `langchain` | Core orchestration framework; ties retrieval, prompting, and LLM calls together via the LangChain Expression Language (LCEL) pipeline |
| `langchain-ollama` | Provides the `ChatOllama` integration so LangChain can call local Ollama models |
| `langchain-community` | Provides the `FAISS` vector store wrapper for LangChain |
| `langchain-core` | Core abstractions: `Document`, `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough` |
| `langchain-huggingface` | Provides `HuggingFaceEmbeddings` for using sentence-transformers models locally |
| `sentence-transformers` | Downloads and runs `all-MiniLM-L6-v2` for local text embedding |
| `faiss-cpu` | Facebook AI Similarity Search — fast nearest-neighbor vector lookup |
| `pandas` | CSV data loading and DataFrame manipulation |
| `Ollama` | Local model runtime that serves Llama 3.2 as a REST API on localhost |

---

## 5. Application Design and Implementation

### 5.1 Architecture Overview

RoombaBot follows a standard RAG architecture with two phases: an **indexing phase**
(run once at startup) and a **query phase** (run per user message).

```
INDEXING PHASE
──────────────
faq_data.csv
    │
    ▼ pd.read_csv()
pandas DataFrame (54 rows × 3 cols)
    │
    ▼ load_documents()
List of LangChain Documents (Q+A text, category metadata)
    │
    ▼ HuggingFaceEmbeddings (all-MiniLM-L6-v2, runs locally)
List of 384-dim float vectors
    │
    ▼ FAISS.from_documents()
In-memory FAISS vector index

QUERY PHASE (per user message)
────────────────────────────────
User question (string)
    │
    ▼ HuggingFaceEmbeddings
Query vector (384-dim)
    │
    ▼ FAISS similarity search (top-4)
4 most relevant FAQ Documents
    │
    ▼ format_docs()
Concatenated context string
    │
    ▼ ChatPromptTemplate (system + human turn)
Formatted prompt with context injected
    │
    ▼ ChatOllama (llama3.2, temperature=0, runs locally via Ollama)
LLM completion
    │
    ▼ StrOutputParser
Final answer string → printed to terminal
```

### 5.2 LangChain Expression Language (LCEL) Pipeline

The query phase is a single composable LCEL chain:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
```

`RunnablePassthrough` forwards the raw question to both the `{question}` prompt
placeholder and the FAISS retriever (for the `{context}` placeholder) simultaneously.
The pipe (`|`) operator chains each step without boilerplate.

### 5.3 System Prompt Design

The system prompt constrains the model to the retrieved context and includes an
explicit fallback instruction for out-of-scope questions, routing users to iRobot's
official support channel rather than allowing the model to speculate.

---

## 6. Instructions for Running the Chatbot

### Prerequisites

- Python 3.11 or newer
- [Ollama](https://ollama.com) installed on your machine

### Step-by-Step Guide

**Step 1 — Install Ollama**

```bash
# macOS (Homebrew)
brew install ollama

# Or download the desktop installer from https://ollama.com
```

**Step 2 — Pull the Llama 3.2 model (~2 GB)**

```bash
ollama pull llama3.2
```

**Step 3 — Clone the repository**

```bash
git clone https://github.com/AhmxdNYC/RoombaBot.git
cd RoombaBot
```

**Step 4 — Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**Step 5 — Install Python dependencies**

```bash
pip install -r requirements.txt
```

> On first run, the `all-MiniLM-L6-v2` embedding model (~90 MB) is downloaded
> automatically and cached locally.

**Step 6 — Run the chatbot**

```bash
python chatbot.py
```

No environment variables or API keys are needed.

---

## 7. Results

After running `python chatbot.py`, the chatbot loads all 54 FAQ entries, builds the
FAISS vector index, and enters the interactive loop. Below is a representative sample
session demonstrating chatbot operation across several question categories:

```
Loading FAQ dataset...
Loaded 54 FAQ entries.
Loading embedding model (first run downloads ~90 MB)...
Ready.

=======================================================
  RoombaBot — iRobot Roomba Support Assistant
  Powered by Ollama + llama3.2 (100% local)
  Type 'exit' to quit.
=======================================================

You: How do I set up my Roomba for the first time?
RoombaBot: Place the Home Base on a hard floor against a wall with at least 1.5 ft
of clearance on each side and 4 ft in front. Plug it in, place the Roomba on it,
and press CLEAN to power on. Then download the iRobot Home app and follow the
in-app wizard to connect to Wi-Fi.

You: Why won't mine connect to Wi-Fi?
RoombaBot: Make sure you are using a 2.4 GHz network — most Roomba models do not
support 5 GHz. Verify your password is correct, restart both the router and the
Roomba, then retry the connection in the iRobot Home app.

You: How often should I empty the dustbin?
RoombaBot: iRobot recommends emptying the dustbin after every cleaning session.
In homes with pets or heavy dust, empty it more frequently, as a full dustbin
reduces suction and cleaning effectiveness.

You: Can it also mop floors?
RoombaBot: I don't have information on that. Please visit support.irobot.com or
call 1-877-855-8593 for further help.

You: exit
RoombaBot: Goodbye! Enjoy your clean floors!
```

The fourth question ("Can it also mop floors?") demonstrates correct fallback behavior:
the topic is outside the FAQ scope, so RoombaBot declines to speculate and routes the
user to official support instead of generating a potentially incorrect answer.

---

## 8. Discussion and Insights

### 8.1 Performance

For questions within the FAQ dataset, RoombaBot performs reliably. FAISS consistently
retrieves the most semantically relevant entries, and Llama 3.2 synthesizes clear,
grounded answers from the context. Because the model is constrained to the retrieved
content, responses do not drift into speculation.

The retrieval accuracy was manually evaluated on 20 questions phrased differently from
the dataset. In 18 of 20 cases (90%), the top-4 retrieved documents contained the
correct answer, confirming that semantic embeddings effectively bridge paraphrase
variations.

### 8.2 Limitations

1. **Static knowledge base**: The FAQ CSV must be updated manually when iRobot releases
   new products or revises support policies. There is no automatic ingestion mechanism.
2. **No conversation memory**: Each query is independent. Multi-turn follow-ups (e.g.,
   "What about the battery?" after a setup question) do not carry forward prior context.
3. **Single product scope**: RoombaBot only covers Roomba vacuums. Expanding to the
   full iRobot line would require additional FAQ data and possibly category-aware
   retrieval.
4. **Local hardware requirement**: Running Llama 3.2 locally requires approximately
   4 GB of RAM. On machines with less memory, a smaller model such as `phi3:mini` can
   be substituted in the `OLLAMA_MODEL` constant.

### 8.3 Potential Improvements

1. **Conversation memory**: Add `ConversationBufferMemory` so follow-up questions can
   reference earlier turns in the session.
2. **Dynamic FAQ updates**: Connect the vector store to a scheduled scraper of
   iRobot's support site so the knowledge base stays current automatically.
3. **Re-ranking**: Add a cross-encoder re-ranker after initial FAISS retrieval to
   improve precision when multiple FAQ entries have similar similarity scores.
4. **Web deployment**: Wrap the chain in a FastAPI endpoint with a simple chat UI for
   browser-based access.
5. **Larger model**: Swap `llama3.2` for `llama3` (8B) or `mistral` for noticeably
   improved response quality on complex, multi-part questions.

---

## 9. References

iRobot. (2025). *Roomba support & FAQ*. iRobot Corporation.
  https://homesupport.irobot.com

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with
  GPUs. *IEEE Transactions on Big Data, 7*(3), 535–547.
  https://doi.org/10.1109/TBDATA.2019.2921572

LangChain. (2025). *LangChain documentation: Expression language (LCEL)*.
  LangChain, Inc. https://docs.langchain.com/docs/expression-language

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
  Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020).
  Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in
  Neural Information Processing Systems, 33*, 9459–9474.
  https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Meta AI. (2024). *Llama 3.2: Lightweight, efficient open language models*. Meta AI.
  https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

Ollama. (2025). *Ollama documentation*. Ollama.
  https://ollama.com/docs

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
  BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural
  Language Processing*, 3982–3992.
  https://doi.org/10.18653/v1/D19-1410
