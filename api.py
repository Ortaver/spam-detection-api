from flask import Flask, request, jsonify, render_template_string
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import traceback

app = Flask(__name__)

pipeline = joblib.load("models/tfidf.pkl")
word_vec = pipeline['word_vec']
char_vec  = pipeline['char_vec']
selector  = pipeline['selector']

nb  = joblib.load("models/nb.pkl")
svm = joblib.load("models/svm_hybrid.pkl")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Email Detector</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 25px 45px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #fff;
            font-size: 28px;
            margin-bottom: 8px;
        }
        .header p {
            color: rgba(255,255,255,0.6);
            font-size: 14px;
        }
        .badge {
            display: inline-block;
            background: linear-gradient(90deg, #e94560, #0f3460);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-bottom: 15px;
        }
        textarea {
            width: 100%;
            height: 180px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 15px;
            color: #fff;
            font-size: 14px;
            resize: vertical;
            outline: none;
            transition: border 0.3s;
        }
        textarea:focus {
            border-color: rgba(233,69,96,0.6);
        }
        textarea::placeholder { color: rgba(255,255,255,0.3); }
        button {
            width: 100%;
            padding: 15px;
            margin-top: 15px;
            background: linear-gradient(90deg, #e94560, #c23152);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.3s, transform 0.1s;
        }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            display: none;
        }
        .result.spam {
            background: rgba(233,69,96,0.15);
            border: 1px solid rgba(233,69,96,0.4);
        }
        .result.ham {
            background: rgba(39,174,96,0.15);
            border: 1px solid rgba(39,174,96,0.4);
        }
        .result-label {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .result.spam .result-label { color: #e94560; }
        .result.ham .result-label { color: #27ae60; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 15px;
        }
        .metric {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #fff;
        }
        .metric-label {
            font-size: 11px;
            color: rgba(255,255,255,0.5);
            margin-top: 4px;
        }
        .loading {
            text-align: center;
            color: rgba(255,255,255,0.6);
            margin-top: 15px;
            display: none;
        }
        .footer {
            text-align: center;
            margin-top: 25px;
            color: rgba(255,255,255,0.3);
            font-size: 12px;
        }
        .progress-bar {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        .spam .progress-fill { background: #e94560; }
        .ham .progress-fill { background: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge">MSc Research — JSTU Makurdi 2026</div>
            <h1>🛡️ Spam Email Detector</h1>
            <p>Feature-Level Hybrid Naïve Bayes–SVM Model</p>
        </div>

        <textarea id="emailText" placeholder="Paste your email text here to classify it as spam or ham..."></textarea>

        <button onclick="classify()" id="btn">Analyse Email</button>

        <div class="loading" id="loading">⏳ Analysing email...</div>

        <div class="result" id="result">
            <div class="result-label" id="resultLabel"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width:0%"></div>
            </div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" id="confidence">—</div>
                    <div class="metric-label">SVM Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="spamProb">—</div>
                    <div class="metric-label">NB Spam Probability</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="hamProb">—</div>
                    <div class="metric-label">NB Ham Probability</div>
                </div>
            </div>
        </div>

        <div class="footer">
            ADAA, Richard Tavershima · 19/12610/MSc · Computer Science
        </div>
    </div>

    <script>
        async function classify() {
            const text = document.getElementById('emailText').value.trim();
            if (!text) { alert('Please enter some email text.'); return; }

            const btn = document.getElementById('btn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            btn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                const isSpam = data.prediction === 'spam';
                const confidence = (data.confidence * 100).toFixed(1);
                const spamProb = (data.nb_probabilities.spam * 100).toFixed(1);
                const hamProb = (data.nb_probabilities.ham * 100).toFixed(1);

                result.className = 'result ' + data.prediction;
                document.getElementById('resultLabel').innerHTML =
                    isSpam ? '🚨 SPAM DETECTED' : '✅ LEGITIMATE EMAIL (Ham)';
                document.getElementById('confidence').textContent = confidence + '%';
                document.getElementById('spamProb').textContent = spamProb + '%';
                document.getElementById('hamProb').textContent = hamProb + '%';
                document.getElementById('progressFill').style.width = confidence + '%';

                result.style.display = 'block';

            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }

        document.getElementById('emailText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') classify();
        });
    </script>
</body>
</html>
"""

def preprocess_for_api(email_text):
    X_word = word_vec.transform([email_text])
    X_char = char_vec.transform([email_text])
    X = hstack([X_word, X_char])
    X = selector.transform(X)
    return X


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    email_text = data.get("text", "").strip()

    if not email_text:
        return jsonify({"error": "Email text cannot be empty"}), 400

    try:
        X = preprocess_for_api(email_text)
        probs = nb.predict_proba(X)
        log_probs = np.log(probs + 1e-9)
        probs_sparse     = csr_matrix(probs)
        log_probs_sparse = csr_matrix(log_probs)
        X_hybrid = hstack([X, probs_sparse, log_probs_sparse])
        pred  = svm.predict(X_hybrid)[0]
        label = "spam" if pred == 1 else "ham"
        confidence = svm.predict_proba(X_hybrid)[0][1]

        return jsonify({
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "nb_probabilities": {
                "ham":  round(float(probs[0][0]), 4),
                "spam": round(float(probs[0][1]), 4)
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=False)