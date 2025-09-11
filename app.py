from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 🔹 Replace with your Hugging Face Space URL
HF_API_URL = "https://<your-username>-pixpop-sdxl-lcm.hf.space/run/predict"

@app.route("/")
def home():
    return "✅ Pixpop Render backend is live (proxy to Hugging Face Space)."

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "A futuristic city at sunset, ultra realistic, 8k")
        steps = int(data.get("steps", 8))
        guidance = float(data.get("guidance", 1.0))

        # 🔹 Forward request to Hugging Face Space API
        response = requests.post(
            HF_API_URL,
            json={"data": [prompt, steps, guidance]},  # order must match Gradio function
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            # Usually result["data"][0] is the image (base64 or URL)
            return jsonify({"status": "ok", "result": result})
        else:
            return jsonify({"status": "error", "details": response.text}), 500

    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
