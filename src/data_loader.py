import os

def load_data(base_path):
    texts = []
    labels = []

    ham_path = os.path.join(base_path, "ham")
    spam_path = os.path.join(base_path, "spam")

    # Load ham (label = 0)
    for file in os.listdir(ham_path):
        with open(os.path.join(ham_path, file), 'r', encoding='latin-1') as f:
            texts.append(f.read())
            labels.append(0)

    # Load spam (label = 1)
    for file in os.listdir(spam_path):
        with open(os.path.join(spam_path, file), 'r', encoding='latin-1') as f:
            texts.append(f.read())
            labels.append(1)

    return texts, labels