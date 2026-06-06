# RoombaBot — iRobot Roomba FAQ Chatbot

An FAQ chatbot that answers questions about iRobot Roomba robot vacuums using a
Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Ollama, and FAISS.
**Runs 100% locally — no API key or account required.**

## Files

| File | Description |
|---|---|
| `chatbot.py` | Main application source code |
| `faq_data.csv` | 54-entry FAQ dataset (curated from iRobot official support docs) |
| `requirements.txt` | Python dependencies |
| `report.md` | Full assignment report |

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed on your machine

## Setup

**Step 1 — Install Ollama and pull the model**

```bash
# macOS (via Homebrew)
brew install ollama

# Or download the installer from https://ollama.com

ollama pull llama3.2
```

**Step 2 — Create a virtual environment and install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> The first run also downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) automatically.

## Run

Make sure Ollama is running (it usually starts automatically after install), then:

```bash
python chatbot.py
```

## Example Session

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

You: How long does the battery last?
RoombaBot: Most Roomba models run 60–120 minutes per charge. The i-series and
j-series typically last about 75 minutes, while the s9+ runs up to 120 minutes.
The robot returns to the Home Base automatically when the battery gets low.

You: Why won't mine connect to Wi-Fi?
RoombaBot: Verify you are using a 2.4 GHz network — most Roomba models do not
support 5 GHz. Check that your Wi-Fi password is correct and restart both the
router and the Roomba, then retry the connection in the iRobot Home app.

You: exit
RoombaBot: Goodbye! Enjoy your clean floors!
```
