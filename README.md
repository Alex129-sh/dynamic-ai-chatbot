# Dynamic AI Chatbot — Complete Flask Project

A Flask portfolio / college project that combines:

- OpenAI-compatible LLM (Groq by default)
- Lightweight NLP (intent, NER, VADER sentiment)
- SQLite conversation history, feedback & analytics
- User-controlled long-term memory
- Multilingual replies (English / Hindi / Hinglish)
- Browser voice input & text-to-speech
- Multi-format document Q&A (PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT)

## Features vs. project brief

| Requirement | Status |
|-------------|--------|
| NLP intent recognition & NER | Yes – keyword + regex patterns |
| Contextual / long-term memory | Yes – session + explicit "Remember that…" |
| Generative AI responses | Yes – Groq (or any OpenAI-compatible API) |
| Sentiment analysis | Yes – VADER |
| Analytics dashboard | Yes – /dashboard |
| Multilingual | Yes – Auto / English / Hindi / Hinglish |
| Voice | Yes – Web Speech API (Chrome/Edge) |
| Document Q&A | Yes – multi-file, page/slide/sheet labels |
| Multi-platform (WhatsApp etc.) | API-ready; web UI shipped |

## Project layout

```
.
├── app.py                 # Flask routes & document extraction
├── chatbot/
│   ├── database.py        # SQLite layer
│   ├── nlp.py             # Sentiment, intent, entities
│   └── llm.py             # Groq / OpenAI reply generation
├── templates/
│   ├── index.html         # Chat UI
│   └── dashboard.html     # Analytics
├── static/style.css
├── tests/
├── instance/              # SQLite DB + uploaded text cache
├── requirements.txt
├── .env.example
└── README.md
```

## Supported uploads

PDF, DOCX, TXT, CSV, XLSX, XLS, PPTX. Multiple files can be selected together.
Spreadsheet uploads include computed missing/duplicate checks, numeric summaries, medians and correlations.

Limits: scanned/image-only PDFs need OCR (not bundled); first 5 000 rows per sheet; max 10 MB per file; max ~10 documents per browser session.

## Run locally

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env → set GROQ_API_KEY and a strong FLASK_SECRET_KEY
python app.py
```

Open http://127.0.0.1:5000 — Dashboard: http://127.0.0.1:5000/dashboard

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env
python app.py
```

## Validation

```bash
python -m unittest discover -s tests -v
```

Manually verify: text chat, new chat / history, feedback, memory add/delete, each file type, multi-file selection, PDF page references, CSV/Excel mean/median questions.

## Environment variables

| Variable | Purpose |
|----------|---------|
| GROQ_API_KEY | Required for AI replies |
| GROQ_MODEL | Default openai/gpt-oss-20b |
| GROQ_BASE_URL | Default Groq OpenAI-compatible URL |
| FLASK_SECRET_KEY | Session signing |
| PORT | Default 5000 |
| CHATBOT_DB_PATH | Optional custom SQLite path |

## Deployment notes

Use a host that supports Flask/Python and persistent disk. Prefer gunicorn app:app on Linux. Configure HTTPS, auth, rate limits and malware scanning before public exposure. Never commit .env.
