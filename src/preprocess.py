import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack

lemmatizer = WordNetLemmatizer()

ENGLISH_STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'get', 'got', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
    'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such',
    't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'us', 'very', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}


def clean_text(text):
    """
    Apply text cleaning pipeline:
    1. Remove email headers
    2. Remove HTML tags
    3. Remove URLs and email addresses
    4. Remove special characters
    5. Lowercase
    6. Tokenise, remove stop words, lemmatise
    """
    text = re.sub(r'^(From|To|Subject|Cc|Bcc|Date|Message-ID|MIME|Content)[^\n]*\n',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def extract_features(texts, labels):
    """
    Stratified 80:20 split — 26,972 train / 6,744 test.

    All vectorisers and the chi-squared selector are fitted on the
    training partition only and applied to the test partition via
    transform, preventing any vocabulary or statistical leakage.

    Decision threshold tuning is performed separately on out-of-fold
    predictions from the training partition (see evaluate.py); the
    test partition is never consulted during threshold selection.
    """
    print("Cleaning and lemmatising texts...")
    texts = [clean_text(str(t)) for t in texts]

    # ── Stratified 80:20 split ────────────────────────────────────────────────
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    # ~26,972 train  /  ~6,744 test

    # ── Word-level TF-IDF (unigrams + bigrams, 6,000 features) ───────────────
    word_vec = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_word = word_vec.fit_transform(texts_train)
    X_test_word  = word_vec.transform(texts_test)

    # ── Character-level TF-IDF (3–5 char n-grams, 4,000 features) ───────────
    char_vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=4000,
        sublinear_tf=True
    )
    X_train_char = char_vec.fit_transform(texts_train)
    X_test_char  = char_vec.transform(texts_test)

    # ── Concatenate → 10,000 dimensions ──────────────────────────────────────
    X_train = hstack([X_train_word, X_train_char])
    X_test  = hstack([X_test_word,  X_test_char])

    # ── Chi-squared selection → top 7,000 features ───────────────────────────
    selector = SelectKBest(chi2, k=7000)
    X_train  = selector.fit_transform(X_train, y_train)
    X_test   = selector.transform(X_test)

    print(f"Feature matrix — Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, (word_vec, char_vec, selector)


def save_pipeline(word_vec, char_vec, selector, path="models/tfidf.pkl"):
    import joblib, os
    os.makedirs("models", exist_ok=True)
    joblib.dump({'word_vec': word_vec, 'char_vec': char_vec,
                 'selector': selector}, path)
    print(f"Pipeline saved to {path}")