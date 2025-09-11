from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)
CORS(app)

# Hugging Face Space API endpoint
HF_API_URL = "https://huggingface.co/spaces/karthikn11/pixpop.hf.space/run/predict"

# Hugging Face token stored in Railway Variables
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route("/", methods=["GET"])
def home():
    return "✅ Pixpop Railway backend is live"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json or {}
        prompt = data.get("prompt", "A futuristic city at sunset")
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance", 1.0))

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        # Send request to Hugging Face Space
        response = requests.post(
            HF_API_URL,
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
