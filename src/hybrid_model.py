import numpy as np
from scipy.sparse import hstack

def build_hybrid_features(nb_model, X_train, X_test):
    # NB probabilities (dense but small)
    train_probs = nb_model.predict_proba(X_train)
    test_probs = nb_model.predict_proba(X_test)

    # Log probabilities
    train_log = np.log(train_probs + 1e-9)
    test_log = np.log(test_probs + 1e-9)

    # Convert small arrays to sparse
    from scipy.sparse import csr_matrix
    train_probs = csr_matrix(train_probs)
    test_probs = csr_matrix(test_probs)

    train_log = csr_matrix(train_log)
    test_log = csr_matrix(test_log)

    # ✅ KEEP EVERYTHING SPARSE
    X_train_hybrid = hstack([X_train, train_probs, train_log])
    X_test_hybrid = hstack([X_test, test_probs, test_log])

    return X_train_hybrid, X_test_hybrid