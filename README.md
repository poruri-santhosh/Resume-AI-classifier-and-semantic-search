# Resume AI classifier and semantic search

A Flask website and JSON API that loads a Hugging Face resume dataset, embeds
its text with the PyTorch-backed `all-MiniLM-L6-v2` Sentence Transformer,
trains a scikit-learn `LogisticRegression` classifier, and indexes embeddings
in an in-memory Qdrant collection. PostgreSQL stores prediction categories,
confidence scores, and timestamps; submitted resume text is not stored.

## 1. Prerequisites

- Python 3.10 or newer (Python 3.11 is used by Docker).
- Docker Desktop, if using the included PostgreSQL setup.
- Access to the Hugging Face dataset configured by `HF_DATASET_NAME`.

The default dataset ID is exactly `C0ldSmi1eresume-dataset`. Confirm it exists
and that your account can read it. For a private or gated dataset, create a
Hugging Face read token and set `HF_TOKEN`; otherwise set `HF_DATASET_NAME` to
an accessible resume dataset. The dataset needs resume-text and category
fields. If automatic field detection does not match its schema, set
`TEXT_COLUMN` and `LABEL_COLUMN`.

## 2. Configure

Copy `.env.example` to `.env` and edit the dataset settings if needed. In
PowerShell:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env` or put a Hugging Face token in source control.

## 3. Run the full stack with Docker

```powershell
docker compose up --build
```

Open <http://localhost:5000>. The first classify or search request loads the
dataset and model, trains and indexes the data, and may take several minutes.
PostgreSQL is started automatically. Stop the containers with `Ctrl+C`; remove
the local database volume only if you intentionally want to discard its stored
prediction metadata.

## 4. Run locally (optional)

Start PostgreSQL first (for example, run the `db` service from
`docker-compose.yml`), then:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Set `DATABASE_URL` to a PostgreSQL connection string reachable from the local
machine. If it is omitted, prediction logging is disabled; classification and
search still work.

## 5. API

`POST /api/classify-resume`

```json
{"text": "Experienced data analyst with Python and SQL experience"}
```

Returns the best category, its confidence, and up to three ranked category
alternatives.

`POST /api/search`

```json
{"text": "Python data analyst", "limit": 5}
```

Returns up to 20 nearest dataset records, with similarity score, category, and
a short excerpt. `limit` defaults to 5.

`GET /api/health` reports whether the model is loaded and whether PostgreSQL
logging is enabled. A successful health response does not trigger the initial
model and dataset load.

## Notes

- The Qdrant collection is intentionally in-memory and is rebuilt when the
  process restarts.
- The first request initializes the model and dataset lazily to keep module
  import and the health endpoint responsive. An initialization failure is
  logged and returned as HTTP 503 with configuration guidance.
- The application uses the dataset's `train` split by default. Set
  `HF_DATASET_SPLIT` to use another available split.
- Limit memory/startup time with `MAX_DATASET_ROWS`; `0` means all usable rows.
- This is a demo without user authentication. Do not expose it publicly without
  adding authentication, rate limits, and a privacy review.
