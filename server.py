# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# Change this to your HF API URL (for deployed HF Space or local testing)
HF_API_URL = "http://127.0.0.1:7860/api/predict/"

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "A futuristic city at sunset")
        
        response = requests.post(HF_API_URL, json={"data":[prompt]}, timeout=120)

        if response.status_code == 200:
            return jsonify({"status":"ok", "result": response.json()})
        else:
            return jsonify({"status":"error", "details": response.text}), 500
    except Exception as e:
        return jsonify({"status":"error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
