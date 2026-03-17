import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import hstack

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from enron_preprocessing import load_enron_dataset

# =====================================
# Create Output Folders
# =====================================

os.makedirs("Models", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# =====================================
# Load Dataset
# =====================================

df = load_enron_dataset("Enron/spam", "Enron/ham")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# =====================================
# TF-IDF Feature Extraction
# =====================================

tfidf = TfidfVectorizer(stop_words="english")

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# =====================================
# Train Naïve Bayes
# =====================================

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# =====================================
# NB Probability Features
# =====================================

nb_train_probs = nb_model.predict_proba(X_train_tfidf)
nb_test_probs = nb_model.predict_proba(X_test_tfidf)

# =====================================
# Hybrid Feature Construction
# =====================================

X_train_hybrid = hstack([X_train_tfidf, nb_train_probs])
X_test_hybrid = hstack([X_test_tfidf, nb_test_probs])

# =====================================
# Train SVM
# =====================================

svm_model = LinearSVC()
svm_model.fit(X_train_hybrid, y_train)

# =====================================
# Prediction
# =====================================

hybrid_pred = svm_model.predict(X_test_hybrid)

# =====================================
# Evaluation Metrics (FIXED)
# =====================================

accuracy = accuracy_score(y_test, hybrid_pred)
precision = precision_score(y_test, hybrid_pred, average="weighted")
recall = recall_score(y_test, hybrid_pred, average="weighted")
f1 = f1_score(y_test, hybrid_pred, average="weighted")

print("\nHybrid NB–SVM Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}\n")

print("Hybrid NB–SVM Classification Report")

print(
    classification_report(
        y_test,
        hybrid_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        digits=4,
        zero_division=0
    )
)

# =====================================
# Confusion Matrix (Improved)
# =====================================

cm = confusion_matrix(y_test, hybrid_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot(cmap="Blues", values_format='d')

plt.title("Hybrid NB–SVM Confusion Matrix")
plt.savefig("Results/hybrid_confusion_matrix.png")
plt.show()

# =====================================
# Save Results Table
# =====================================

results = pd.DataFrame({
    "Model": ["Hybrid NB-SVM"],
    "Accuracy": [round(accuracy, 4)],
    "Precision": [round(precision, 4)],
    "Recall": [round(recall, 4)],
    "F1 Score": [round(f1, 4)]
})

results.to_csv("Results/hybrid_results.csv", index=False)

print("\nResults table saved to Results/hybrid_results.csv")

# =====================================
# Save Models for API Deployment
# =====================================

joblib.dump(tfidf, "Models/hybrid_tfidf.pkl")
joblib.dump(nb_model, "Models/hybrid_nb.pkl")
joblib.dump(svm_model, "Models/hybrid_svm.pkl")

print("\nHybrid model saved successfully.")