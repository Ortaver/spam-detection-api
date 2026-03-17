import os
import re
import nltk
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ========================================
# Download NLTK Resources (First Run Only)
# ========================================

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# =========================================
# Text Preprocessing (Lemmatization)
# =========================================

def lemmatize_text(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmas)

# =========================================
# Load Dataset
# =========================================

def load_emails(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                content = f.read()
                data.append((content, label))
    return data

ham_data = load_emails("Enron/ham", 0)
spam_data = load_emails("Enron/spam", 1)

df = pd.DataFrame(ham_data + spam_data, columns=["text", "label"])

print("Dataset size:", df.shape)

# Apply Lemmatization
print("Applying lemmatization...")
df["text"] = df["text"].apply(lemmatize_text)

# =========================================
# Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# =========================================
# Cross-Validation Setup
# =========================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create folders
os.makedirs("Models", exist_ok=True)
os.makedirs("Results", exist_ok=True)

# ======================================================
# NAÏVE BAYES MODEL
# ======================================================

print("\n===== Training Naïve Bayes =====")

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB())
])

nb_param_grid = {
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__min_df": [1, 2],
    "nb__alpha": [0.1, 0.5, 1.0]
}

nb_grid = GridSearchCV(
    nb_pipeline,
    nb_param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

nb_grid.fit(X_train, y_train)

print("Best NB Parameters:", nb_grid.best_params_)
print(f"Best NB CV Score: {nb_grid.best_score_:.4f}")

nb_best = nb_grid.best_estimator_

nb_pred = nb_best.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)

print(f"\nNaïve Bayes Test Accuracy: {nb_accuracy:.4f}")

print("\nNaïve Bayes Classification Report")

print(
    classification_report(
        y_test,
        nb_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        digits=4,
        zero_division=0
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    nb_pred,
    display_labels=["Ham", "Spam"]
)

plt.title("Naïve Bayes Confusion Matrix")
plt.show()

joblib.dump(nb_best, "Models/nb_model.pkl")

# ======================================================
# SVM MODEL
# ======================================================

print("\n===== Training SVM =====")

svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("svm", LinearSVC())
])

svm_param_grid = {
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__min_df": [1, 2],
    "svm__C": [0.1, 1, 10]
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("Best SVM Parameters:", svm_grid.best_params_)
print(f"Best SVM CV Score: {svm_grid.best_score_:.4f}")

svm_best = svm_grid.best_estimator_

svm_pred = svm_best.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"\nSVM Test Accuracy: {svm_accuracy:.4f}")

print("\nSVM Classification Report")

print(
    classification_report(
        y_test,
        svm_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        digits=4,
        zero_division=0
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    svm_pred,
    display_labels=["Ham", "Spam"]
)

plt.title("SVM Confusion Matrix")
plt.show()

joblib.dump(svm_best, "Models/svm_model.pkl")

# ======================================================
# SAVE COMPARISON RESULTS
# ======================================================

results = pd.DataFrame({
    "Model": ["Naïve Bayes", "SVM"],
    "Accuracy": [
        round(nb_accuracy, 4),
        round(svm_accuracy, 4)
    ]
})

results.to_csv("Results/individual_results.csv", index=False)

print("\nResults saved to Results/individual_results.csv")

print("\nTraining completed successfully.")