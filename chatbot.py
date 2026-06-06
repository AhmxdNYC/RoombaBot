"""
RoombaBot — FAQ Chatbot for iRobot Roomba
==========================================
LangChain + OpenAI GPT-3.5-turbo + FAISS RAG pipeline.

Usage:
    export OPENAI_API_KEY="sk-..."
    python chatbot.py
"""

import os
import sys
import pandas as pd

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Configuration ──────────────────────────────────────────────────────────────
FAQ_CSV     = "faq_data.csv"
LLM_MODEL   = "gpt-3.5-turbo"
EMBED_MODEL = "text-embedding-3-small"
TOP_K       = 4          # number of FAQ chunks retrieved per query

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
    embeddings  = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    def format_docs(retrieved: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in retrieved)

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[ERROR] OPENAI_API_KEY environment variable is not set.")
    if not os.path.exists(FAQ_CSV):
        sys.exit(f"[ERROR] Dataset not found: {FAQ_CSV}")

    print("Loading FAQ dataset and building vector index...")
    docs  = load_documents(FAQ_CSV)
    chain = build_chain(docs)
    print(f"Ready — {len(docs)} FAQs indexed.\n")

    print("=" * 55)
    print("  RoombaBot — iRobot Roomba Support Assistant")
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
