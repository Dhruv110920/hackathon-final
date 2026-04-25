from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)
CORS(app)

# ================= SAFE MODEL LOADING =================
model = None
scaler = None

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Model Loaded")

except Exception as e:
    print("❌ Model Load Error:", e)


# ================= GEMINI CONFIG =================
API_KEY = "AIzaSyD8g_xNCWUWlr7kZCvoKKj_6B-JutUf9uw"   # keep your working key

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/shipment")
def shipment():
    return render_template("shipment.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/health")
def health():
    return {"status": "ok"}


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        df = pd.DataFrame([data])
        pred = model.predict_proba(scaler.transform(df))[0][1]

        return jsonify({"delay_probability": round(pred * 100, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= GEMINI =================
@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20
        )

        result = response.json()

        print("Gemini Response:", result)

        # HANDLE ERROR RESPONSE SAFELY
        if "error" in result:
            return jsonify({"error": result["error"].get("message", "API error")}), 500

        # SAFE EXTRACTION
        reply = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response")
        )

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= START =================
if __name__ == "__main__":
    app.run(debug=True)