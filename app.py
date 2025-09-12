from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

# 1. Create Flask app
app = Flask(__name__)
CORS(app)

# 2. Hugging Face info
HF_API_URL = "https://huggingface.co/spaces/karthikn11/pixpop.hf.space/run/predict"  # replace with your HF Space URL
HF_TOKEN = os.getenv("HF_TOKEN")  # set this in Railway variables

# 3. Define route AFTER app is created
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "Hello world")

        payload = {"data": [prompt]}
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        response = requests.post(HF_API_URL, json=payload, headers=headers, timeout=60)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"status": "error", "code": response.status_code, "message": response.text}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# 4. For local debugging only
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
