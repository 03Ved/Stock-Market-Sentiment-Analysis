"""Small Gemini REST client with a configurable model and bounded requests."""
import os
import requests


def generate(contents, system_prompt=""):
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    payload = {"contents": contents}
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key}, json=payload, timeout=45,
    )
    if not response.ok:
        raise RuntimeError(f"Gemini returned HTTP {response.status_code}; check your model, key and quota")
    parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text:
        raise RuntimeError("Gemini returned no text")
    return text


def generate_text(prompt):
    return generate([{"role": "user", "parts": [{"text": prompt}]}])
