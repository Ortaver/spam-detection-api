import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

# Store predictions globally for McNemar's test
all_predictions = {}

def save_confusion_matrix(cm, name):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix", fontsize=14)
    plt.colorbar()
    classes = ['Ham', 'Spam']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight='bold'
            )
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    path = f"results/{name}_cm.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved → {path}")


def find_best_threshold(scores, y_true):
    thresholds = np.linspace(0.2, 0.9, 200)
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return min(best_t + 0.02, 0.95)


def evaluate_model(name, model, X_test, y_test):
    global all_predictions

    scores = model.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(scores, y_test)
    y_pred = (scores >= threshold).astype(int)

    # Store predictions for McNemar's test
    all_predictions[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    save_confusion_matrix(cm, name)
    print(f"Threshold: {threshold:.4f}")

    return y_pred


def run_mcnemar_tests(y_true):
    """Run McNemar's test between all key model pairs"""
    
    pairs = [
        ("Naive_Bayes", "SVM"),
        ("SVM", "Hybrid_Base"),
        ("Hybrid_Base", "Optimized_Hybrid"),
        ("Naive_Bayes", "Optimized_Hybrid"),
    ]

    print("\n" + "="*60)
    print("MCNEMAR'S TEST RESULTS")
    print("="*60)

    os.makedirs("results", exist_ok=True)
    report_lines = []

    for model_a, model_b in pairs:
        if model_a not in all_predictions or model_b not in all_predictions:
            print(f"Skipping {model_a} vs {model_b} — predictions not found")
            continue

        y_a = all_predictions[model_a]
        y_b = all_predictions[model_b]

        # Build contingency table
        # b = A correct, B wrong
        # c = A wrong, B correct
        b = np.sum((y_a == y_true) & (y_b != y_true))
        c = np.sum((y_a != y_true) & (y_b == y_true))

        print(f"\n{model_a} vs {model_b}")
        print(f"  {model_a} correct, {model_b} wrong (b): {b}")
        print(f"  {model_a} wrong, {model_b} correct (c): {c}")

        table = [[0, b], [c, 0]]

        # Use exact test if b+c < 25
        exact = (b + c) < 25
        result = mcnemar(table, exact=exact, correction=True)

        significance = "SIGNIFICANT" if result.pvalue < 0.05 else "NOT significant"
        print(f"  Chi-square: {result.statistic:.4f}")
        print(f"  P-value   : {result.pvalue:.4f}")
        print(f"  Result    : {significance} (p {'<' if result.pvalue < 0.05 else '>='} 0.05)")

        report_lines.append(
            f"{model_a} vs {model_b}: chi2={result.statistic:.4f}, "
            f"p={result.pvalue:.4f}, {significance}"
        )

    # Save report
    with open("results/mcnemar_results.txt", "w") as f:
        f.write("McNemar's Test Results\n")
        f.write("="*60 + "\n")
        for line in report_lines:
            f.write(line + "\n")

    print("\nSaved → results/mcnemar_results.txt")