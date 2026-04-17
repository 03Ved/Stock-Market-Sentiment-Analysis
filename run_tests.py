import os
import subprocess
import time
import requests

def run_tests():
    # Start API server
    print("Starting API Server...")
    api_proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        env=dict(os.environ, PYTHONPATH=".")
    )
    time.sleep(4)

    try:
        print("\n--- Testing Health ---")
        r = requests.get("http://localhost:8000/health")
        print(f"Status: {r.status_code}")
        print(r.json())

        print("\n--- Testing Search ---")
        r = requests.get("http://localhost:8000/search?q=reliance")
        print(f"Status: {r.status_code}")
        print(r.json())

        print("\n--- Testing Ticker Data ---")
        r = requests.get("http://localhost:8000/ticker/RELIANCE?window=7d")
        print(f"Status: {r.status_code}")
        print(r.json())

        print("\n--- Testing Ticker History ---")
        r = requests.get("http://localhost:8000/ticker/RELIANCE/history?hours=168")
        print(f"Status: {r.status_code}")
        print("Got points:", len(r.json().get('points', [])))

        print("\n--- Testing Live Fetch Fallback ---")
        r = requests.get("http://localhost:8000/ticker/IDEA?window=30d")
        print(f"Status: {r.status_code}")
        print(r.json())
        
    finally:
        api_proc.terminate()
        print("API Server dead")

if __name__ == "__main__":
    run_tests()
