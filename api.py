from flask import Flask, request, jsonify
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load models and full preprocessing pipeline
pipeline = joblib.load("models/tfidf.pkl")
word_vec = pipeline['word_vec']
char_vec  = pipeline['char_vec']
selector  = pipeline['selector']

nb  = joblib.load("models/nb.pkl")
svm = joblib.load("models/svm_hybrid.pkl")


def preprocess_for_api(email_text):
    X_word = word_vec.transform([email_text])
    X_char = char_vec.transform([email_text])
    X = hstack([X_word, X_char])
    X = selector.transform(X)
    return X


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Spam Detection API is live",
        "usage": "POST /predict with JSON body: {\"text\": \"your email text here\"}"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    email_text = data.get("text", "").strip()

    if not email_text:
        return jsonify({"error": "Email text cannot be empty"}), 400

    try:
        # Step 1: Extract features
        X = preprocess_for_api(email_text)

        # Step 2: NB probabilities
        probs = nb.predict_proba(X)

        # Step 3: Log probabilities
        log_probs = np.log(probs + 1e-9)

        # Step 4: Build hybrid feature vector
        probs_sparse     = csr_matrix(probs)
        log_probs_sparse = csr_matrix(log_probs)
        X_hybrid = hstack([X, probs_sparse, log_probs_sparse])

        # Step 5: SVM prediction
        pred  = svm.predict(X_hybrid)[0]
        label = "spam" if pred == 1 else "ham"

        # Step 6: Confidence score
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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)