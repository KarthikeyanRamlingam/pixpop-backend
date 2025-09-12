from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Allow requests from anywhere (Netlify frontend)

# Use your Hugging Face token here (set in Railway Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_API_URL = "https://karthikn11-pixpop.hf.space/run/predict
"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route("/")
def home():
    return "✅ Pixpop Railway backend is live."

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "A futuristic city at sunset, ultra realistic, 8k")

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=120
        )
        response.raise_for_status()
        return jsonify({"status": "ok", "result": response.json()})
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

