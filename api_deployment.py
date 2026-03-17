from flask import Flask, request, jsonify
import joblib
from scipy.sparse import hstack
import os

# =====================================
# Initialize Flask App
# =====================================

app = Flask(__name__)

# =====================================
# Load Trained Models (Safe Loading)
# =====================================

MODEL_PATH = "Models"

try:
    tfidf = joblib.load(os.path.join(MODEL_PATH, "hybrid_tfidf.pkl"))
    nb_model = joblib.load(os.path.join(MODEL_PATH, "hybrid_nb.pkl"))
    svm_model = joblib.load(os.path.join(MODEL_PATH, "hybrid_svm.pkl"))
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    tfidf, nb_model, svm_model = None, None, None

# =====================================
# Home Route
# =====================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Hybrid NB-SVM Spam Detection API is running"
    })

# =====================================
# Health Check Route (VERY IMPORTANT)
# =====================================

@app.route("/health", methods=["GET"])
def health():
    if tfidf and nb_model and svm_model:
        return jsonify({"status": "healthy"})
    else:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500

# =====================================
# Prediction Route
# =====================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if models loaded
        if not (tfidf and nb_model and svm_model):
            return jsonify({
                "status": "error",
                "message": "Models not available"
            }), 500

        data = request.get_json()

        # Validate input
        if not data or "email" not in data:
            return jsonify({
                "status": "error",
                "message": "Please provide 'email' field in JSON body"
            }), 400

        text = data["email"]

        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                "status": "error",
                "message": "Invalid email content"
            }), 400

        # =====================================
        # Feature Processing
        # =====================================

        text_tfidf = tfidf.transform([text])
        nb_probs = nb_model.predict_proba(text_tfidf)
        hybrid_features = hstack([text_tfidf, nb_probs])

        # =====================================
        # Prediction
        # =====================================

        prediction = svm_model.predict(hybrid_features)[0]
        decision_score = svm_model.decision_function(hybrid_features)[0]

        label = "Spam" if prediction == 1 else "Ham"

        # Normalize confidence (optional but nicer)
        confidence = float(abs(decision_score))

        # =====================================
        # Response
        # =====================================

        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence_score": confidence
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =====================================
# Run Server (Render Compatible)
# =====================================
@app.route("/test", methods=["GET"])
def test():
    return {
        "message": "API is working correctly 🚀",
        "status": "success"
    }
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)