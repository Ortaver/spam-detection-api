from flask import Flask, request, jsonify, render_template_string
import joblib
from scipy.sparse import hstack
import os

app = Flask(__name__)

# =============================
# Load Models
# =============================

MODEL_PATH = os.path.join(os.getcwd(), "Models")
print("Current working directory:", os.getcwd())
print("Looking for models in:", MODEL_PATH)
try:
    tfidf = joblib.load(os.path.join(MODEL_PATH, "hybrid_tfidf.pkl"))
    nb_model = joblib.load(os.path.join(MODEL_PATH, "hybrid_nb.pkl"))
    svm_model = joblib.load(os.path.join(MODEL_PATH, "hybrid_svm.pkl"))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    tfidf, nb_model, svm_model = None, None, None

# =============================
# Home Route
# =============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "success", "message": "Spam Detection API Running"})

# =============================
# Web Application (Professional UI)
# =============================

@app.route("/app", methods=["GET", "POST"])
def web_app():
    result = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            text = request.form.get("email")

            if not text:
                error = "No input provided"

            elif not (tfidf and nb_model and svm_model):
                error = "Models not loaded on server"

            else:
                text_tfidf = tfidf.transform([text])
                nb_probs = nb_model.predict_proba(text_tfidf)
                hybrid_features = hstack([text_tfidf, nb_probs])

                prediction = svm_model.predict(hybrid_features)[0]
                decision_score = svm_model.decision_function(hybrid_features)[0]

                result = "Spam" if prediction == 1 else "Ham"
                confidence = round(float(abs(decision_score)), 4)

        except Exception as e:
            error = str(e)

    return render_template_string("""
    <html>
    <head>
        <title>Spam Detection</title>
    </head>
    <body style="font-family: Arial; text-align: center; margin-top: 50px;">

        <h2>📧 Spam Detection System</h2>

        <form method="POST">
            <textarea name="email" rows="6" cols="50" placeholder="Enter email text..."></textarea><br><br>
            <button type="submit">Check</button>
        </form>

        {% if result %}
            <h3>Prediction: {{ result }}</h3>
            <h4>Confidence: {{ confidence }}</h4>
        {% endif %}

        {% if error %}
            <h3 style="color:red;">Error: {{ error }}</h3>
        {% endif %}

    </body>
    </html>
    """, result=result, confidence=confidence, error=error)

# =============================
# Run App
# =============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
