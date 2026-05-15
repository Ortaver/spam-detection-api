from src.data_loader import load_data
from src.preprocess import extract_features
from src.train_models import train_nb, train_svm
from src.hybrid_model import build_hybrid_features
from src.optimize import optimize_svm
from src.evaluate import evaluate_model, run_mcnemar_tests

def main():
    print("Loading data...")
    texts, labels = load_data("enron")

    print("Extracting features...")
    X_train, X_test, y_train, y_test, vecs = extract_features(texts, labels)

    print("Training NB...")
    nb = train_nb(X_train, y_train)

    print("Training SVM...")
    svm = train_svm(X_train, y_train)

    print("Building Hybrid...")
    X_train_hybrid, X_test_hybrid = build_hybrid_features(nb, X_train, X_test)

    print("Training Hybrid...")
    hybrid = train_svm(X_train_hybrid, y_train)

    print("Optimizing Hybrid...")
    optimized, params = optimize_svm(X_train_hybrid, y_train)

    print("\n--- Evaluation ---")
    evaluate_model("Naive_Bayes", nb, X_test, y_test)
    evaluate_model("SVM", svm, X_test, y_test)
    evaluate_model("Hybrid_Base", hybrid, X_test_hybrid, y_test)
    evaluate_model("Optimized_Hybrid", optimized, X_test_hybrid, y_test)

    print("\n--- McNemar's Test ---")
    run_mcnemar_tests(y_test)

    print("\n=== Done ===")

if __name__ == "__main__":
    main()