"""
Exercise 1 — Basic LLM App
User -> App -> API -> Ollama -> Code Llama -> Response

A minimal Flask app with one /ask route that forwards a question straight
to Code Llama via Ollama's local HTTP API. No knowledge base yet — this
just proves the plumbing works end to end.
"""
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama"


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": question,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Could not reach Ollama. Is it running? Try: ollama serve"
        }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ollama request failed: {e}"}), 502

    answer = response.json().get("response", "")
    return jsonify({"question": question, "answer": answer})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Port 5000 collides with macOS AirPlay Receiver on some Macs -> use 5001.
    app.run(host="0.0.0.0", port=5001, debug=True)
