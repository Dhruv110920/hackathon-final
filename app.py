from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)
CORS(app)

# =========================
# SAFE MODEL LOADING
# =========================
model = None
scaler = None

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("✅ Model & Scaler Loaded Successfully")

except Exception as e:
    print("❌ Model Load Error:", e)


# =========================
# GEMINI API
# =========================
API_KEY = "AIzaSyCiDIODwiF-luAQdXKIyNEA_tr1S_b1ofc"   # ← replace with your working key


# =========================
# WEATHER API FUNCTION
# =========================
def get_weather_temp(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url, timeout=5)
        data = res.json()

        return data["current_weather"]["temperature"]

    except Exception as e:
        print("Weather API Error:", e)
        return 25  # fallback temp


# =========================
# FEATURE ENGINEERING
# =========================
FEATURES = [
    "latitude", "longitude", "inventory_level", "temperature",
    "traffic_status", "transaction_amount", "purchase_frequency",
    "delay_reason", "asset_utilization", "demand_forecast",
    "logistics_delay", "transport_mode",
    "is_freezing", "is_extreme_cold", "is_cold", "is_hot", "is_extreme_heat", "is_normal_temp",
    "traffic_high", "traffic_medium", "traffic_low",
    "weather_traffic", "road_traffic", "logistics_traffic",
    "demand_inv_ratio", "weather_logistics", "heat_logistics", "temp_abs", "has_delay_reason",
]

def add_features(df):
    df = df.copy()
    df["is_freezing"]       = (df["temperature"] < 0).astype(int)
    df["is_extreme_cold"]   = (df["temperature"] < -10).astype(int)
    df["is_cold"]           = ((df["temperature"] >= -10) & (df["temperature"] < 0)).astype(int)
    df["is_hot"]            = (df["temperature"] > 35).astype(int)
    df["is_extreme_heat"]   = (df["temperature"] > 42).astype(int)
    df["is_normal_temp"]    = ((df["temperature"] >= 5) & (df["temperature"] <= 32)).astype(int)
    df["traffic_high"]      = (df["traffic_status"] == 2).astype(int)
    df["traffic_medium"]    = (df["traffic_status"] == 1).astype(int)
    df["traffic_low"]       = (df["traffic_status"] == 0).astype(int)
    df["weather_traffic"]   = df["temperature"].abs() * df["traffic_status"]
    df["road_traffic"]      = (df["transport_mode"] == 0).astype(int) * df["traffic_status"]
    df["logistics_traffic"] = df["logistics_delay"] * df["traffic_status"]
    df["demand_inv_ratio"]  = df["demand_forecast"] / (df["inventory_level"] + 1)
    df["weather_logistics"] = df["is_freezing"] * df["logistics_delay"]
    df["heat_logistics"]    = df["is_hot"] * df["logistics_delay"]
    df["temp_abs"]          = df["temperature"].abs()
    df["has_delay_reason"]  = (df["delay_reason"] > 0).astype(int)
    return df


# =========================
# ROUTES (FRONTEND)
# =========================
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


# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        lat = float(data["latitude"])
        lon = float(data["longitude"])

        # 🔥 GET LIVE WEATHER
        temp = get_weather_temp(lat, lon)

        df = pd.DataFrame([{
            "latitude": lat,
            "longitude": lon,
            "inventory_level": int(data["inventory_level"]),
            "temperature": temp,
            "traffic_status": int(data["traffic_status"]),
            "transaction_amount": int(data["transaction_amount"]),
            "purchase_frequency": int(data["purchase_frequency"]),
            "delay_reason": int(data["delay_reason"]),
            "asset_utilization": float(data["asset_utilization"]),
            "demand_forecast": int(data["demand_forecast"]),
            "logistics_delay": int(data["logistics_delay"]),
            "transport_mode": int(data.get("transport_mode", 0)),
        }])

        df = add_features(df)[FEATURES]

        prob = model.predict_proba(scaler.transform(df))[0][1]

        risk = "Low" if prob < 0.35 else "Moderate" if prob < 0.65 else "High"

        return jsonify({
            "delay_probability": round(prob * 100, 2),
            "risk": risk,
            "temperature_used": temp
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# GEMINI AI ROUTE
# =========================
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

        if "error" in result:
            return jsonify({"error": result["error"]["message"]}), 500

        reply = result.get("candidates", [{}])[0] \
                      .get("content", {}) \
                      .get("parts", [{}])[0] \
                      .get("text", "No response")

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# START
# =========================
if __name__ == "__main__":
    app.run(debug=True)