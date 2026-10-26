# Validation — 27 October 2026

## Verified locally

- Python 3.12.6 on Windows, CPU-only PyTorch 2.6.0 and Transformers 4.44.2.
- `python -m pytest -q`: **18 passed**. Covers API status/validation, search, ticker linking, SQLite deduplication and confidence flags, weighting and confidence gating, history bucket boundaries and UTC timestamps, in-process/HTTP clients, Streamlit rendering and input changes, missing-key behavior, mocked Google News retry/deduplication, and Gemini payload/error handling.
- All application Python files parsed successfully; the original explainer syntax error is fixed.
- Actual public ProsusAI/finbert download and inference: the headline “Reliance Industries reports record quarterly profit and strong growth” returned score **+0.9281**, confidence **0.9500**. This is a smoke test, not a predictive accuracy benchmark.
- Actual Integrated Gradients execution (8 interpolation steps) returned finite word attributions. The explainer shares the scorer's checkpoint and explains a fixed predicted class.
- Actual Google News ingestion inserted **24 scored articles** for Reliance. The 7-day ticker endpoint found **5 articles**, score **-0.0048**, confidence **low**. Live figures change over time. The original verification command encountered a Windows console encoding error while printing a rupee sign after the data checks; this did not affect ingestion or aggregation.

## Streamlit Community Cloud

- Deployed from `03Ved/Stock-Market-Sentiment-Analysis`, branch `main`, entrypoint `dashboard/app.py`.
- Public URL: https://indian-stock-sentiment-analysis.streamlit.app/
- Verified Python 3.12.15, Streamlit 1.56.0, CPU-only PyTorch 2.6.0 in build logs.
- Hosted app fetched news on its empty database, loaded FinBERT, and rendered Reliance sentiment (-0.005), five recent articles, and the history/source charts.
- Original repository and deployment fork both contain the implementation commit `02b2c75`.

## Limits

- The fine-tuned checkpoint is absent. Historical model metrics were not reproduced. Evaluation mixes gold and silver labels; the stored accuracy/F1 is not an independent benchmark.
- No real credentialed Gemini, GNews, or MarketAux request was tested. Gemini request construction and errors were tested with mocks.
- RSS/Reddit/NDTV scheduler integrations are present but not all upstream sources were live-tested. The Streamlit demo uses on-demand Google News; it does not start the scheduler.
- Streamlit instance storage is ephemeral; cold starts and upstream availability affect demos.
