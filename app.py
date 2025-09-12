@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    return jsonify({
        "status": "ok",
        "prompt_received": data.get("prompt")
    })
