from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Health check route
@app.route("/")
def home():
    return "✅ Backend is live"

# Main generate route
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "A futuristic city at sunset, ultra realistic, 8k")
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance", 1.0))

        # Here you should call HF API or simulate response
        # For now, let's just return the prompt to test
        return jsonify({"status": "ok", "prompt_received": prompt})

    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
