"""
Exercise 4 — LLM Service (port 5004)

Thin wrapper around Ollama's local API. Every other service talks to
Code Llama only through this one, over HTTP, instead of calling Ollama
directly.
"""
import os

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("MODEL_NAME", "codellama")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    num_predict = int(data.get("num_predict", 150))
    model = data.get("model") or MODEL_NAME

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": num_predict, "num_ctx": 1024},
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Could not reach Ollama. Is it running? Try: ollama serve"}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 502

    return jsonify({"response": response.json().get("response", "")})


@app.route("/health", methods=["GET"])
def health():
    try:
        tags_url = OLLAMA_URL.replace("/api/generate", "/api/tags")
        requests.get(tags_url, timeout=5).raise_for_status()
        ollama_ok = True
    except requests.exceptions.RequestException:
        ollama_ok = False
    return jsonify({"status": "ok", "ollama_reachable": ollama_ok, "model": MODEL_NAME})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True, use_reloader=False)
