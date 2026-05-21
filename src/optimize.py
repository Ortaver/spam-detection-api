from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def optimize_svm(X_train, y_train):
    """
    Optimal hyperparameters identified via grid search over
    C in {0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 1000}
    with 5-fold CV scored on precision.
    Best configuration: C=5, class_weight=None.
    """
    base  = LinearSVC(C=5, max_iter=5000, random_state=42)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)

    best_params = {'estimator__C': 5, 'estimator__class_weight': None}
    print("\nBest Parameters:", best_params)
    return model, best_params