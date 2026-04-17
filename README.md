# Stock-Market-Sentiment-Analysis

This project is a comprehensive market sentiment analysis system for Indian stocks (NSE/BSE). It collects financial news and headlines from various sources including RSS feeds, Google News, MarketAux, NDTV, and Reddit, processes them using NLP models like FinBERT, and provides real-time sentiment scores through a FastAPI backend and a Streamlit dashboard.

## Features

- **Data Ingestion**: Automated scraping from multiple news sources and social media
- **Sentiment Analysis**: Uses fine-tuned FinBERT model for Indian market context
- **Real-time API**: RESTful endpoints for sentiment data and search
- **Interactive Dashboard**: Streamlit app for visualization and analysis
- **Database Storage**: SQLite-based storage with efficient querying
- **Evaluation Tools**: Model performance metrics and comparison

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-market-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional, for API keys):
   Create a `.env` file in the root directory with:
   ```
   MARKETAUX_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   GNEWS_API_KEY=your_key_here
   ```

### Usage

1. **Start the API server**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Run the dashboard** (in a separate terminal):
   ```bash
   streamlit run dashboard/app.py
   ```

3. **Data Ingestion** (optional, for collecting new data):
   - Run the scheduler for automated ingestion:
     ```bash
     python ingestion/scheduler.py
   ```
   - Or run individual scrapers:
     ```bash
     python ingestion/google_news_scraper.py
     python ingestion/reddit_scraper.py
     # etc.
   ```

4. **Test the API**:
   ```bash
   ./test_endpoints.sh
   ```

### API Endpoints

- `GET /health` - Health check
- `GET /search?q=<query>` - Search documents
- `GET /ticker/{symbol}?window=<window>` - Get sentiment for a ticker
- `GET /ticker/{symbol}/history?hours=<hours>` - Get historical sentiment data
- `GET /explain/{doc_id}` - Get explanation for a document

### Dashboard Features

- Real-time sentiment charts
- Ticker search and analysis
- Model evaluation metrics
- AI-powered sentiment assistance

## Project Structure

- `api/` - FastAPI backend
- `dashboard/` - Streamlit frontend
- `ingestion/` - Data collection scripts
- `nlp/` - Sentiment analysis and NLP processing
- `models/` - Trained models and checkpoints
- `db/` - Database schema and queries
- `evaluation/` - Model evaluation and plotting
- `config/` - Configuration settings