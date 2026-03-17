import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def load_enron_dataset(spam_path, ham_path):
    def read_emails(path):
        emails = []
        for root, _, files in os.walk(path):
            for file in files:
                with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                    emails.append(f.read())
        return emails

    spam_emails = read_emails(spam_path)
    ham_emails = read_emails(ham_path)

    df = pd.DataFrame({
        'text': ham_emails + spam_emails,
        'label': [0] * len(ham_emails) + [1] * len(spam_emails)
    })

    return df

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def preprocess_and_split(df):
    df['text'] = df['text'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test