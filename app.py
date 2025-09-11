from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "✅ Pixpop Railway backend is live"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = data.get("prompt", "default prompt")
    return jsonify({"status": "ok", "prompt_received": prompt})
