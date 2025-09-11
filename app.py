from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Hugging Face Space URL (replace if needed)
HF_SPACE_API = "https://karthikn11-pixpop-sdxl-lcm.hf.space/run/predict"

# HF token stored in Railway Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")

@app.route("/")
def home():
    return "✅ Pixpop Railway backend is live (proxy to Hugging Face Space)."

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json or {}
        prompt = data.get("prompt", "A futuristic city at sunset")
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance", 1.0))

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        # POST request to HF Space
        response = requests.post(
            HF_SPACE_API,
            headers=headers,
            json={"data": [prompt, steps, guidance]},
            timeout=120
        )

        if response.status_code == 200:
            return jsonify({"status": "ok", "result": response.json()})
        else:
            return jsonify({"status": "error", "details": response.text}), 500

    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
