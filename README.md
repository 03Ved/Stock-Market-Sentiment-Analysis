# Stock Market Sentiment Analysis — Indian Equities

An NLP application that fetches financial news, links companies to an NSE symbol master, scores headlines with FinBERT, and combines article scores using source reliability, model confidence, and exponential time decay. Streamlit provides ticker search, sentiment charts, source articles, Integrated Gradients explanations, saved evaluation results, and an optional Gemini assistant. FastAPI exposes the same analysis separately.

## Run the demo

Use **Python 3.12 (64-bit)**. Run commands from this repository's root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.
Only Streamlit is needed. The dashboard calls backend functions directly by default; a separate API process is optional. The database is initialized automatically. Search starts with Reliance. On an empty database, Google News RSS is fetched and scored on demand, with no API key required. The initial request downloads ProsusAI/finbert (roughly 440 MB); subsequent requests reuse the model. The dashboard caches ticker responses for 15 minutes. The refresh button clears that cache; fresh articles are fetched when stored coverage is insufficient.

News availability depends on the upstream provider. If fewer than two scored articles fall inside the selected window, the app reports insufficient data. A chart bucket also requires two articles. Try a longer window when coverage is sparse. No synthetic articles or fabricated sentiment scores are inserted.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository you own, retaining attribution and any applicable upstream terms.
2. Open https://share.streamlit.io/ and choose **Create app**.
3. Select your repository and branch. Set the main file to **dashboard/app.py**.
4. In **Advanced settings**, choose **Python 3.12**. Runtime dependencies are in root `requirements.txt`; the Linux wheel is CPU-only and targets Community Cloud's Python 3.12 environment.
5. Optional: paste the needed entries from `.streamlit/secrets.toml.example` into **Secrets**. Leave `API_BASE_URL` unset for standalone hosting.
6. Deploy, then test Reliance, another ticker, multiple windows, article links, and the explanation button. Add the verified public URL to your resume.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Streamlit top-level secrets become environment variables. Never commit `.env`, `.streamlit/secrets.toml`, credentials, or model binaries. The SQLite database and model cache are local to the hosted instance; they can be lost on restart or redeployment. The app rebuilds its database and fetches news on demand. Community Cloud may sleep while idle, so a cold start can require downloading/loading FinBERT again. The scheduler is not started by the hosted dashboard. This demo is on-demand analysis, not a continuously running ingestion service.

## Optional integrations

Copy `.env.example` to `.env` locally, or use Streamlit Secrets in the cloud:

- `GEMINI_API_KEY`: enables the chat assistant.
- `GEMINI_MODEL`: model available to your account (default `gemini-3.8-flash`); change it when provider availability changes.
- `GNEWS_API_KEY`: adds GNews headlines.
- `MARKETAUX_API_KEY`: enables a secondary news fallback.
- `API_BASE_URL`: only set this if using a separately hosted FastAPI backend.
- `DB_PATH`: optional database location.

Without a Gemini key the assistant shows a configuration message; the dashboard still works. Provider credentials, quota, and model access must be checked separately before claiming these optional features work on your deployment. Optional AI news query/relevance enrichment is disabled unless `ENABLE_GEMINI_ENRICHMENT=1`.

## Optional API and background ingestion

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
python -m ingestion.scheduler
```

These are separate processes for local use or infrastructure you manage. To connect the dashboard to the API, set `API_BASE_URL=http://localhost:8000` locally; cloud hosting requires a reachable HTTPS API URL.

- `GET /health`: database counts.
- `GET /search?q=reliance`: matching NSE symbols and aliases.
- `GET /ticker/RELIANCE?window=7d`: weighted sentiment and top articles.
- `GET /ticker/RELIANCE/history?hours=168&resolution=auto`: time buckets.
- `GET /explain/{doc_id}`: token attribution for a stored article headline.

Supported windows: 1h, 6h, 12h, 24h, 3d, 7d, 30d, and custom hours up to 8760h. Sentiment ranges from -1 to +1 and is not a prediction of returns. Source weights are manually chosen. The tracked NSE master and aliases are snapshots and need periodic refresh; independent BSE-only symbol support is not implemented.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary databases and provider mocks. They cover window parsing, ticker linking, duplicate inserts, confidence flags, weighting, API validation, history buckets, local/external clients, dashboard rendering, and optional-key behavior. They do not prove every news provider or a credentialed Gemini account works. See `VALIDATION.md` for the most recent checks and limits.

## Model and evaluation caveats

The deployed runtime uses pretrained **ProsusAI/finbert** unless a local checkpoint exists at `models/finbert-indian-markets/best/`. That checkpoint is absent from this checkout and model binaries are gitignored. Fine-tuning code is supplied in `nlp/fine_tune.py`; training is not required for the demo.

`evaluation/model_comparison.json` and plots are historical artifacts. Their provenance has not been independently verified. The comparison script mixes manually labelled and FinBERT-generated silver labels before splitting. This makes its scores unsuitable as an independent accuracy estimate. In the saved results pretrained macro F1 (0.9264) exceeds fine-tuned macro F1 (0.8979). Do not claim the fine-tuning improved performance or cite those numbers as verified accuracy on your resume. Use an independently annotated, held-out test set for a defensible benchmark.

A resume description after you have reviewed and understood your contribution:

> Developed a Streamlit financial-news sentiment dashboard for Indian equities using FinBERT, NSE ticker linking, SQLite storage, source-weighted time-decay aggregation, and token-level explanations, with a FastAPI interface and automated integration tests.

Only add deployment, performance improvements, or independently measured metrics after validating them. If this is based on an existing project, credit the original work and describe your own changes accurately.

## Structure

- `ingestion/`: RSS, Google News, MarketAux, NDTV, Reddit, scheduler.
- `nlp/`: FinBERT inference, ticker linking, baseline, fine-tuning, explanations.
- `db/`: SQLite schema, inserts, queries.
- `aggregation/`: source/confidence/time weighted sentiment.
- `api/`: FastAPI endpoints.
- `dashboard/`: Streamlit UI and shared in-process/HTTP data client.
- `agent/`: optional Gemini chat and bounded REST client.
- `evaluation/`: evaluation scripts and historical artifacts.
- `tests/`: deterministic validation.
