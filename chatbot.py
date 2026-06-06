"""
RoombaBot — FAQ Chatbot for iRobot Roomba
==========================================
LangChain + Ollama (llama3.2) + FAISS RAG pipeline.
Runs entirely locally — no API key required.

Prerequisites:
    1. Install Ollama:  https://ollama.com
    2. Pull the model:  ollama pull llama3.2
    3. Keep Ollama running in the background (it starts automatically on most systems)

Usage:
    python chatbot.py
"""

import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Configuration ──────────────────────────────────────────────────────────────
FAQ_CSV      = "faq_data.csv"
OLLAMA_MODEL = "llama3.2"        # change to "llama3" for the 8B model
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # ~90 MB, auto-downloaded
TOP_K        = 4

SYSTEM_PROMPT = (
    "You are RoombaBot, a friendly and knowledgeable customer support assistant "
    "for iRobot Roomba robot vacuums. Answer the user's question using ONLY the "
    "FAQ context provided below. Be concise and accurate. "
    "If the answer cannot be found in the context, respond with: "
    "'I don\\'t have information on that. Please visit support.irobot.com or call "
    "1-877-855-8593 for further help.'\n\n"
    "FAQ Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={"category": str(row.get("category", "General"))},
        ))
    return docs


# ── RAG Pipeline ───────────────────────────────────────────────────────────────
def build_chain(docs: list[Document]):
    print("Loading embedding model (first run downloads ~90 MB)...")
    embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    def format_docs(retrieved: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in retrieved)

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    if not __import__("os").path.exists(FAQ_CSV):
        sys.exit(f"[ERROR] Dataset not found: {FAQ_CSV}")

    print(f"Loading FAQ dataset...")
    docs = load_documents(FAQ_CSV)
    print(f"Loaded {len(docs)} FAQ entries.")

    chain = build_chain(docs)
    print(f"Ready.\n")

    print("=" * 55)
    print("  RoombaBot — iRobot Roomba Support Assistant")
    print("  Powered by Ollama + llama3.2 (100% local)")
    print("  Type 'exit' to quit.")
    print("=" * 55 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRoombaBot: Goodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "bye", "q"}:
            print("RoombaBot: Goodbye! Enjoy your clean floors!")
            break

        answer = chain.invoke(question)
        print(f"RoombaBot: {answer}\n")


if __name__ == "__main__":
    main()
