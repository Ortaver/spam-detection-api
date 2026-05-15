from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def optimize_svm(X_train, y_train):
    base = LinearSVC(max_iter=5000)
    model = CalibratedClassifierCV(base, cv=3)

    param_grid = {
        'estimator__C': [1, 5, 10],
        'estimator__class_weight': [None, 'balanced']
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:", grid.best_params_)
    return grid.best_estimator_, grid.best_params_