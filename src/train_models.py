from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def train_nb(X_train, y_train):
    nb = MultinomialNB(alpha=0.3)
    nb.fit(X_train, y_train)
    return nb

def train_svm(X_train, y_train):
    base = LinearSVC(C=1.0, max_iter=5000)
    svm = CalibratedClassifierCV(base, cv=3)  # enables predict_proba
    svm.fit(X_train, y_train)
    return svm