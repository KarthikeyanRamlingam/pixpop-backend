from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_SPACE_API = os.environ.get("HF_SPACE_API")  # we’ll set this in Railway

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt")
    r = requests.post(HF_SPACE_API, json={"data": [prompt]})
    data = r.json()
    # This may need adjusting depending on Hugging Face return format
    image_url = data["data"][0]
    return jsonify({"image_url": image_url})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
