from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HF_API_URL = "https://karthikn11-pixpop.hf.space/api/predict/"

@app.route("/")
def home():
    return "✅ Pixpop Railway backend is live!"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "A futuristic city at sunset, 8k")
        payload = {"data": [prompt]}

        response = requests.post(HF_API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return jsonify({"status": "ok", "result": response.json()})
        else:
            return jsonify({"status": "error", "details": response.text}), 500

    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
