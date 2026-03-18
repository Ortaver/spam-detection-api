from flask import Flask, request, jsonify, render_template_string
import joblib
from scipy.sparse import hstack
import os
import math

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
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    tfidf, nb_model, svm_model = None, None, None

# =============================
# Home Route
# =============================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Spam Detection API Running"
    })

# =============================
# Web Application (Upgraded UI)
# =============================

@app.route("/app", methods=["GET", "POST"])
def web_app():
    result = None
    confidence = None
    percent = None
    error = None

    if request.method == "POST":
        try:
            text = request.form.get("email")

            if not text:
                error = "Please enter email text"

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

                # Convert to percentage
                prob = 1 / (1 + math.exp(-decision_score))
                percent = round(prob * 100, 2)

        except Exception as e:
            error = str(e)

    return render_template_string("""
    <html>
    <head>
        <title>Spam Detection App</title>
        <style>
            body {
                font-family: Arial;
                background: linear-gradient(to right, #4e73df, #1cc88a);
                text-align: center;
                color: white;
            }
            .container {
                background: white;
                color: black;
                width: 50%;
                margin: auto;
                margin-top: 80px;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
            }
            textarea {
                width: 90%;
                padding: 10px;
                border-radius: 8px;
            }
            button {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                background-color: #4e73df;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background-color: #2e59d9;
            }
            #loading {
                display: none;
                margin-top: 10px;
            }
            .result {
                margin-top: 20px;
                font-size: 20px;
                font-weight: bold;
            }
            .spam { color: red; }
            .ham { color: green; }
            .bar {
                width: 100%;
                background: #ddd;
                border-radius: 10px;
                margin-top: 10px;
            }
            .fill {
                height: 20px;
                border-radius: 10px;
            }
        </style>

        <script>
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
            }
        </script>
    </head>

    <body>
        <div class="container">
            <h2>📧 Hybrid NB-SVM Spam Detector</h2>

            <form method="POST" onsubmit="showLoading()">
                <textarea name="email" rows="6" placeholder="Enter email text..."></textarea><br><br>
                <button type="submit">Check</button>
            </form>

            <div id="loading">⏳ Analyzing... Please wait</div>

            {% if result %}
                <div class="result {{ 'spam' if result == 'Spam' else 'ham' }}">
                    Prediction: {{ result }}
                </div>

                <p>Confidence Score: {{ confidence }}</p>
                <p>Confidence: {{ percent }}%</p>

                <div class="bar">
                    <div class="fill"
                         style="width: {{ percent }}%; background: {{ 'red' if result == 'Spam' else 'green' }};">
                    </div>
                </div>
            {% endif %}

            {% if error %}
                <h3 style="color:red;">Error: {{ error }}</h3>
            {% endif %}
        </div>
    </body>
    </html>
    """, result=result, confidence=confidence, percent=percent, error=error)

# =============================
# Run App
# =============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
