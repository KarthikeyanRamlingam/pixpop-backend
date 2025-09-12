from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

HF_API_URL = "https://karthikn11-pixpop.hf.space/run/predict"
HF_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def home():
    return "✅ Pixpop Railway backend is live."

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "A futuristic city at sunset, ultra realistic, 8k")

        payload = {"inputs": prompt}

        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            image_base64 = result[0]['image_base64'] if isinstance(result, list) else None

            return jsonify({
                "status": "ok",
                "prompt_received": prompt,
                "image_base64": image_base64
            })
        else:
            return jsonify({"status": "error", "details": response.text}), 500

    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
