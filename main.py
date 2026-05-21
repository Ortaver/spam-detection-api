from src.data_loader import load_data
from src.preprocess import extract_features, save_pipeline
from src.train_models import train_nb, train_svm
from src.hybrid_model import build_hybrid_features
from src.optimize import optimize_svm
from src.evaluate import evaluate_model, run_mcnemar_tests, print_comparison_table
import joblib


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("Loading data...")
    texts, labels = load_data("enron")

    # ── 2. Preprocess — original 80:20 split ─────────────────────────────────
    print("Extracting features...")
    X_train, X_test, y_train, y_test, vecs = extract_features(texts, labels)
    word_vec, char_vec, selector = vecs
    save_pipeline(word_vec, char_vec, selector)

    # ── 3. Train base models ──────────────────────────────────────────────────
    print("\nTraining Naive Bayes...")
    nb = train_nb(X_train, y_train)

    print("Training standalone SVM...")
    svm = train_svm(X_train, y_train)

    # ── 4. Build hybrid feature matrices ─────────────────────────────────────
    print("Building hybrid feature matrices...")
    X_train_h, X_test_h = build_hybrid_features(nb, X_train, X_test)

    # ── 5. Train base hybrid and optimized hybrid ─────────────────────────────
    print("Training base hybrid SVM...")
    hybrid = train_svm(X_train_h, y_train)

    print("Optimizing hybrid SVM (grid search)...")
    optimized, best_params = optimize_svm(X_train_h, y_train)
    print(f"Best params: {best_params}")

    # ── 6. Save models ────────────────────────────────────────────────────────
    joblib.dump(nb,        "models/nb.pkl")
    joblib.dump(optimized, "models/svm_hybrid.pkl")
    print("Models saved.")

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    # Each call finds the threshold via 5-fold OOF CV on the training set,
    # then applies it once to X_test. The test set is never seen during
    # threshold selection — no optimistic bias.
    print("\n--- Evaluation ---")
    results = []

    results.append(evaluate_model(
        "Naive_Bayes", nb,
        X_train,   y_train,
        X_test,    y_test
    ))
    results.append(evaluate_model(
        "SVM", svm,
        X_train,   y_train,
        X_test,    y_test
    ))
    results.append(evaluate_model(
        "Hybrid_Base", hybrid,
        X_train_h, y_train,
        X_test_h,  y_test
    ))
    results.append(evaluate_model(
        "Optimized_Hybrid", optimized,
        X_train_h, y_train,
        X_test_h,  y_test
    ))

    # ── 8. Table 7 (with FPR % column) ───────────────────────────────────────
    print_comparison_table(results)

# ── 9. McNemar's tests ────────────────────────────────────────────────────
    print("\n--- McNemar's Tests ---")
    run_mcnemar_tests(y_test)


    print("\n=== Done ===")


if __name__ == "__main__":
    main()