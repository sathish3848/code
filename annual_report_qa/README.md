# Annual Report Q&A Assistant (RAG)

Upload a company's annual report (PDF) and ask financial questions in plain English.
Answers are grounded only in the report, with exact page-number citations.

## Architecture

```
PDF Upload → Load Text → Chunk (1000/150) → Embed (Gemini) → ChromaDB
                                                                 ↓
User Question → Embed Query → Retrieve Top-K (k=4) → Prompt + Gemini → Answer + Page Citations
                                                                 ↓
                                                         Streamlit Chat UI
```

## Tech stack

Python, LangChain, ChromaDB, Google Gemini (gemini-3.8-flash), Streamlit, uv

## Features

- PDF upload with automatic indexing
- Page-numbered source citations
- Multi-report support with a report switcher
- Compare Reports tab (same question, two reports, side by side)
- Chat memory, retrieved-context inspector, and chat export to Markdown
- Anti-hallucination prompt: says "I could not find this in the report." when the answer is missing

## Project structure

```
annual-report-qa/
├── app.py            # Streamlit UI
├── config.py         # Settings, models, prompt
├── ingest.py         # Load, chunk, embed, store
├── rag_chain.py      # Retrieve, prompt, Gemini, citations
├── evaluate.py       # 10-question evaluation suite
├── data/             # Uploaded PDFs
├── chroma_db/        # Vector store (auto-created)
├── pyproject.toml    # Dependencies (uv)
├── .env              # GOOGLE_API_KEY (never commit)
└── README.md
```

## Setup

1. Install dependencies: `uv sync`
2. Copy `.env.example` to `.env` and add your Google AI Studio key (starts with `AIza`)
3. Run the app: `uv run streamlit run app.py`
4. Upload a PDF in the sidebar and start asking questions

## Evaluation

Run `uv run python evaluate.py` and paste the generated table below.

| # | Category | Question | Expected Page | Retrieved Pages | Answer (short) | Status |
|---|---|---|---|---|---|---|
| | | | | | | |

## Future improvements

- Hybrid search (keyword + vector)
- Re-ranking of retrieved chunks
- KPI dashboard cards (revenue, profit, EPS)
