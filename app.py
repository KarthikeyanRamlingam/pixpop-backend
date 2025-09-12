from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Hugging Face Space endpoint (your model endpoint)
HF_API_URL = "https://karthikn11-pixpop.hf.space/run/predict"

# ✅ Hugging Face API token (set in Railway → Variables)
HF_TOKEN = os.environ.get("HF_TOKEN")


@app.route("/")
def home():
    return "✅ Pixpop Railway backend is live (proxy to Hugging Face Space)."


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "A futuristic city at sunset, ultra realistic, 8k")
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance", 1.0))

        # ✅ Send request to Hugging Face Space
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
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
    port = int(os.environ.get("PORT", 8080))  # Railway provides $PORT
    app.run(host="0.0.0.0", port=port)
