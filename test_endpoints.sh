#!/bin/bash
echo "--- Testing Health ---"
curl -s http://localhost:8000/health | python3 -m json.tool

echo "--- Testing Search ---"
curl -s "http://localhost:8000/search?q=reliance" | python3 -m json.tool

echo "--- Testing Ticker Data ---"
curl -s "http://localhost:8000/ticker/RELIANCE?window=7d" | python3 -m json.tool

echo "--- Testing Live Fetch Fallback ---"
curl -s "http://localhost:8000/ticker/IDEA?window=30d" | python3 -m json.tool
