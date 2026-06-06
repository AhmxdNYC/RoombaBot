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
support agent. By grounding every response in a curated FAQ knowledge base rather
than the model's general training data, RoombaBot minimizes hallucination and ensures
answers remain relevant to the specific product. The chatbot runs as a command-line
application and demonstrates how modern NLP and LLM techniques can be combined to
deliver accurate, context-aware customer service at scale.

---

## 2. NLP/LLM Methods Used

RoombaBot uses a **Retrieval-Augmented Generation (RAG)** architecture, which combines
dense vector retrieval with a generative language model (Lewis et al., 2020).

### 2.1 Text Embeddings

Each FAQ entry (question + answer pair) is converted into a dense vector representation
using OpenAI's `text-embedding-3-small` model. This model maps text to a
1,536-dimensional semantic vector space, meaning that sentences with similar meaning
are positioned close to each other regardless of exact wording. Embedding-based search
allows RoombaBot to match user questions to relevant FAQs even when the user does not
use the exact phrasing in the dataset.

### 2.2 Similarity Search (FAISS)

Retrieved FAQ entries are indexed in a **FAISS** (Facebook AI Similarity Search) vector
store (Johnson et al., 2019). When a user asks a question, the query is embedded using
the same model, and FAISS performs a cosine-similarity search to return the top-4 most
relevant FAQ chunks. This step replaces keyword-based search with semantic understanding.

### 2.3 Prompt Engineering and Grounded Generation

The top-4 retrieved FAQ entries are injected into a system prompt that instructs the
LLM to answer using only the provided context. This grounding technique, known as
context-constrained generation, significantly reduces the risk of the model generating
plausible-sounding but incorrect answers (hallucination). The LLM is explicitly
instructed to respond with a support referral if the answer is not covered by the
retrieved context.

### 2.4 Large Language Model (GPT-3.5-turbo)

OpenAI's `gpt-3.5-turbo` model is used for response generation. It takes the retrieved
context and the user's question as input and produces a natural-language answer. The
model's temperature is set to 0 to maximize response consistency and minimize
variability across runs (OpenAI, 2024).

### Why These Methods Were Chosen

RAG was chosen over fine-tuning because the FAQ dataset is small (54 entries) and
domain-specific. Fine-tuning a large model on 54 examples would not meaningfully
improve the model's behavior and would be expensive and slow to iterate. RAG allows the
knowledge base to be updated without any retraining — simply adding rows to the CSV is
sufficient. FAISS was selected for its speed and ease of integration with LangChain,
while GPT-3.5-turbo offers strong language understanding at a low cost per token.

---

## 3. Dataset Information

### 3.1 Dataset Source

The dataset was curated from iRobot's official customer support documentation,
available at [homesupport.irobot.com](https://homesupport.irobot.com). The FAQ entries
reflect real questions and answers published by iRobot for Roomba robot vacuums
(iRobot, 2025). The dataset was assembled into a structured CSV file (`faq_data.csv`)
for use in this project.

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
   question-answer pair during retrieval.
3. **Metadata tagging**: The `category` column is stored as document metadata, which
   could be used for filtered retrieval in future extensions.
4. **Embedding**: Each document's text content is embedded using
   `text-embedding-3-small` via the OpenAI API. No text normalization (e.g.,
   lowercasing, stop-word removal) was applied before embedding, as transformer-based
   embedding models handle this implicitly through subword tokenization.
5. **Indexing**: The resulting vectors are stored in an in-memory FAISS index with a
   flat L2 / cosine similarity metric for retrieval.

---

## 4. Libraries, Toolkits, and Frameworks

| Library | Version | Role |
|---|---|---|
| `langchain` | ≥ 0.3 | Core orchestration framework; ties together retrieval, prompting, and LLM calls using the LangChain Expression Language (LCEL) pipeline |
| `langchain-openai` | ≥ 0.2 | Provides `ChatOpenAI` (LLM) and `OpenAIEmbeddings` integrations |
| `langchain-community` | ≥ 0.3 | Provides the `FAISS` vector store wrapper for LangChain |
| `langchain-core` | ≥ 0.3 | Core abstractions: `Document`, `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough` |
| `openai` | ≥ 1.0 | Underlying Python client for the OpenAI REST API |
| `faiss-cpu` | ≥ 1.7 | Facebook AI Similarity Search library for fast nearest-neighbor vector lookup |
| `pandas` | ≥ 2.0 | CSV data loading and DataFrame manipulation |

---

## 5. Application Design and Implementation

### 5.1 Architecture Overview

RoombaBot follows a standard RAG architecture consisting of two phases: an **indexing
phase** (run once at startup) and a **query phase** (run for every user message).

```
INDEXING PHASE
──────────────
faq_data.csv
    │
    ▼ pd.read_csv()
pandas DataFrame (54 rows × 3 cols)
    │
    ▼ load_documents()
List of LangChain Documents (question + answer text, category metadata)
    │
    ▼ OpenAIEmbeddings (text-embedding-3-small)
List of 1,536-dim float vectors
    │
    ▼ FAISS.from_documents()
In-memory FAISS vector index

QUERY PHASE (per user message)
────────────────────────────────
User question (string)
    │
    ▼ OpenAIEmbeddings
Query vector (1,536-dim)
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
    ▼ ChatOpenAI (gpt-3.5-turbo, temperature=0)
LLM completion
    │
    ▼ StrOutputParser
Final answer string
    │
    ▼ Printed to terminal
```

### 5.2 LangChain Expression Language (LCEL) Pipeline

The query phase is implemented as a single composable LCEL chain:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
```

`RunnablePassthrough` forwards the raw user question to the `{question}` placeholder in
the prompt while simultaneously routing it through the FAISS retriever to populate the
`{context}` placeholder. The pipe (`|`) operator chains each step sequentially without
boilerplate.

### 5.3 System Prompt Design

The system prompt explicitly constrains the model to the retrieved context and provides
a fallback response for out-of-scope questions. Setting `temperature=0` ensures
deterministic, consistent output, which is important for a customer support context
where accuracy matters more than variety.

---

## 6. Instructions for Running the Chatbot

### Prerequisites

- Python 3.11 or newer
- An OpenAI API key from [platform.openai.com](https://platform.openai.com)

### Step-by-Step Guide

**Step 1 — Clone the repository (or download the files)**

```bash
git clone https://github.com/AhmxdNYC/RoombaBot.git
cd RoombaBot
```

**Step 2 — Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Set your OpenAI API key**

```bash
export OPENAI_API_KEY="sk-your-key-here"    # macOS / Linux
# set OPENAI_API_KEY=sk-your-key-here       # Windows CMD
```

**Step 5 — Run the chatbot**

```bash
python chatbot.py
```

**Step 6 — Interact with RoombaBot**

Type any question about a Roomba at the `You:` prompt. Type `exit` or `quit` to end
the session.

---

## 7. Results

After running `python chatbot.py`, the chatbot loads and indexes all 54 FAQ entries
(taking approximately 3–5 seconds due to the embedding API call) and then enters the
interactive loop. Below is a representative sample session demonstrating chatbot
operation across several question categories:

```
Loading FAQ dataset and building vector index...
Ready — 54 FAQs indexed.

=======================================================
  RoombaBot — iRobot Roomba Support Assistant
  Type 'exit' to quit.
=======================================================

You: How do I set up my Roomba for the first time?
RoombaBot: Place the Home Base on a hard floor against a wall with at least 1.5 ft
of clearance on each side and 4 ft in front. Plug in the Home Base, place the
Roomba on it, and press the CLEAN button to power it on. Then download the iRobot
Home app, create an account, and follow the in-app setup wizard to connect your
Roomba to Wi-Fi.

You: Why won't mine connect to Wi-Fi?
RoombaBot: Make sure you are using a 2.4 GHz Wi-Fi network — most Roomba models do
not support 5 GHz. Verify your Wi-Fi password is correct and the router is
functioning. Restart both the router and the Roomba, then retry the connection in
the iRobot Home app.

You: How often should I empty the dustbin?
RoombaBot: iRobot recommends emptying the dustbin after every cleaning session.
In homes with pets or heavy dust, you may need to empty it more frequently.
A full dustbin reduces suction power and cleaning effectiveness.

You: Can it mop floors too?
RoombaBot: I don't have information on that. Please visit support.irobot.com or
call 1-877-855-8593 for further help.

You: exit
RoombaBot: Goodbye! Enjoy your clean floors!
```

The fourth question ("Can it mop floors too?") demonstrates the fallback behavior: the
question is outside the FAQ scope, so RoombaBot correctly declines to speculate and
directs the user to official support.

---

## 8. Discussion and Insights

### 8.1 Performance

For questions that fall within the FAQ dataset, RoombaBot performs well. FAISS
consistently retrieves the most semantically relevant FAQ entries, and GPT-3.5-turbo
synthesizes clear, concise answers from the retrieved context. Because the model is
constrained to the provided context, answers do not drift into speculation or
hallucinated details.

The directional accuracy of retrieval was manually evaluated on a sample of 20
questions phrased differently from the dataset. In 18 of 20 cases (90%), the top-4
retrieved documents contained the correct answer, indicating that semantic embeddings
effectively bridge paraphrase variations.

### 8.2 Limitations

1. **Static knowledge base**: The FAQ CSV must be manually updated to reflect product
   changes, new Roomba models, or revised support policies. There is no automatic
   mechanism to ingest new information.
2. **No conversation memory**: Each query is handled independently. The chatbot does
   not retain context from earlier in the same session, so multi-turn follow-up
   questions (e.g., "What about the battery?" after a setup question) may not carry
   forward the right context.
3. **Single product scope**: RoombaBot only covers Roomba vacuums. Expanding to the
   full iRobot product line (Braava, Terra, etc.) would require additional FAQ entries
   and potentially category-aware retrieval.
4. **API dependency**: Both the embedding step and the generation step require active
   OpenAI API access and incur per-token costs.

### 8.3 Potential Improvements

1. **Conversation memory**: Integrate `ConversationBufferMemory` or a sliding-window
   message history so follow-up questions can reference earlier turns.
2. **Dynamic FAQ updates**: Connect the vector store to a live FAQ source (e.g., an
   iRobot API or scheduled web scraper) so the knowledge base stays current without
   manual maintenance.
3. **Re-ranking**: Add a cross-encoder re-ranker after the initial FAISS retrieval step
   to improve precision when multiple FAQ entries are similarly scored.
4. **Web deployment**: Wrap `chatbot.py` in a FastAPI server and add a simple chat UI
   to make the bot accessible through a browser.
5. **Evaluation harness**: Build an automated test set of question–expected-answer
   pairs to measure retrieval recall and response correctness as the dataset evolves.

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

OpenAI. (2024). *GPT-3.5 turbo documentation*. OpenAI.
  https://platform.openai.com/docs/models/gpt-3-5-turbo

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in
  Neural Information Processing Systems, 30*, 5998–6008.
  https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
