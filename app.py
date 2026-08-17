from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "https://langlyr.phyotp.dev",
    "https://langlyr.vercel.app"
]}})

@app.get("/jisho")
def jisho():
    keyword = request.args.get("keyword", "")

    r = requests.get(
        "https://jisho.org/api/v1/search/words",
        params={"keyword": keyword},
        timeout=10,
    )

    return jsonify(r.json())
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
