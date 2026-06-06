# RoombaBot — iRobot Roomba FAQ Chatbot

An FAQ chatbot that answers questions about iRobot Roomba robot vacuums using a
Retrieval-Augmented Generation (RAG) pipeline built with LangChain, OpenAI, and FAISS.

## Files

| File | Description |
|---|---|
| `chatbot.py` | Main application source code |
| `faq_data.csv` | 54-entry FAQ dataset (curated from iRobot official support docs) |
| `requirements.txt` | Python dependencies |
| `report.md` | Full assignment report |

## Requirements

- Python 3.11+
- An OpenAI API key ([platform.openai.com](https://platform.openai.com))

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
export OPENAI_API_KEY="sk-your-key-here"   # Windows: set OPENAI_API_KEY=sk-...
python chatbot.py
```

## Example Session

```
Loading FAQ dataset and building vector index...
Ready — 54 FAQs indexed.

=======================================================
  RoombaBot — iRobot Roomba Support Assistant
  Type 'exit' to quit.
=======================================================

You: How long does the battery last?
RoombaBot: Most Roomba models run 60–120 minutes per charge. The i-series and
j-series typically last about 75 minutes, while the s9+ runs up to 120 minutes.
The robot returns to the Home Base automatically when the battery gets low.

You: exit
RoombaBot: Goodbye! Enjoy your clean floors!
```
