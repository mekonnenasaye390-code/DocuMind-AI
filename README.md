# DocuMind AI

Upload a text-based PDF, ask questions about it, summarize it, or generate a practice exam.
Flask backend + the HTML/CSS/JS frontend in `static/`.

## Project layout

```
documind-ai/
├── app.py              Flask backend (API + static file serving)
├── requirements.txt
├── .env.example        copy to .env
├── README.md
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then set ANTHROPIC_API_KEY (or use Ollama, see below)
python app.py
```

Open http://127.0.0.1:5000. Check http://127.0.0.1:5000/api/health to confirm the AI provider is configured.

### Using Ollama instead of Claude

```bash
ollama pull llama3.1
# in .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

## Run it in production

Documents are kept in process memory, so run **one worker process** and scale with threads:

```bash
# Linux / macOS
gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 app:app

# Windows
waitress-serve --listen=0.0.0.0:8000 --threads=8 app:app
```

Also set `SECRET_KEY`, and `COOKIE_SECURE=true` / `TRUST_PROXY=true` when behind HTTPS and a reverse proxy.

## API

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/upload` | multipart field `file` (PDF) | `{filename, pages, chunks}` |
| POST | `/api/chat` | `{"question": "..."}` | `{answer, sources: [{page, text}]}` |
| POST | `/api/summary` | none | `{result}` |
| POST | `/api/exam` | `{"count": 5, "difficulty": "easy\|medium\|hard"}` | `{result}` |
| POST | `/api/reset` | none | `{ok: true}` |
| GET | `/api/health` | | `{status, provider, configured}` |

Errors are always JSON: `{"error": "message"}` with a matching HTTP status.

## How it works

- **Upload**: validates size, type and page count, extracts text per page with pypdf, and splits it into overlapping chunks.
- **Chat**: finds the most relevant chunks with BM25 (no vector database needed), sends them with page labels to the model, and returns the answer plus the retrieved passages as sources. Short documents are sent whole. The last 3 exchanges are kept so follow-up questions work.
- **Summary / exam**: the whole document is sent if it fits `CONTEXT_CHAR_BUDGET`; longer documents are sampled evenly across their length.
- **Sessions**: each browser gets a signed cookie and one document, kept in memory and dropped after 2 hours idle. Restarting the server clears all documents.
- **Safety**: upload and AI endpoints are rate limited per IP, document text is treated as untrusted data in the prompts, and responses carry CSP and other security headers.

## Limitations

- Scanned or image-only PDFs have no extractable text and are rejected (adding OCR would be the next step).
- The model is asked for plain text because the frontend displays raw text. If you add a Markdown renderer, relax `to_plain_text` and the prompts in `app.py`.
- In-memory storage means no persistence across restarts and no multi-worker scaling. Swap `DocumentStore` for Redis or a database if you need either.
