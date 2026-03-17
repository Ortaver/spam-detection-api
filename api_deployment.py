from flask import Flask, request, jsonify, render_template_string
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
# Health Check Route
# =====================================

@app.route("/health", methods=["GET"])
def health():
    if tfidf and nb_model and svm_model:
        return jsonify({"status": "healthy"})
    else:
        return jsonify({
            "status": "error",
            "message": "Models not loaded"
        }), 500

# =====================================
# Prediction API (POST ONLY)
# =====================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not (tfidf and nb_model and svm_model):
            return jsonify({
                "status": "error",
                "message": "Models not available"
            }), 500

        data = request.get_json()

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

        text_tfidf = tfidf.transform([text])
        nb_probs = nb_model.predict_proba(text_tfidf)
        hybrid_features = hstack([text_tfidf, nb_probs])

        prediction = svm_model.predict(hybrid_features)[0]
        decision_score = svm_model.decision_function(hybrid_features)[0]

        label = "Spam" if prediction == 1 else "Ham"
        confidence = float(abs(decision_score))

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
# SIMPLE WEB INTERFACE (NEW 🔥)
# =====================================

@app.route("/app", methods=["GET", "POST"])
def web_app():
    result = None

    if request.method == "POST":
        text = request.form.get("email")

        if tfidf and nb_model and svm_model and text:
            text_tfidf = tfidf.transform([text])
            nb_probs = nb_model.predict_proba(text_tfidf)
            hybrid_features = hstack([text_tfidf, nb_probs])

            prediction = svm_model.predict(hybrid_features)[0]
            result = "Spam" if prediction == 1 else "Ham"

    return render_template_string("""
        <html>
        <head>
            <title>Spam Detection App</title>
        </head>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h2>📧 Spam Detection System</h2>
            
            <form method="POST">
                <textarea name="email" rows="5" cols="40" placeholder="Enter email text here..."></textarea><br><br>
                <button type="submit">Check</button>
            </form>

            {% if result %}
                <h3>Prediction: {{ result }}</h3>
            {% endif %}
        </body>
        </html>
    """, result=result)

# =====================================
# Test Route
# =====================================

@app.route("/test", methods=["GET"])
def test():
    return {
        "message": "API is working correctly 🚀",
        "status": "success"
    }

# =====================================
# Run Server (Render Compatible)
# =====================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
