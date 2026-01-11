from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)



@app.route("/predict-risk", methods=["POST"])
def predict_risk():
    data = request.get_json()
    df = pd.DataFrame([data])
    df = df[feature_names]
    pred = model.predict(df)[0]
    proba = model.predict_proba(df).max()
    risk_label = label_encoder.inverse_transform([pred])[0]

    return jsonify({
        "risk_level": risk_label,
        "confidence": round(float(proba), 2),
        "message": f"Client classified as {risk_label} risk"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)