import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack

# Download required NLTK resources on first run
# Run once manually if needed:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

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
    1. Remove HTML tags
    2. Remove email headers (lines starting with From:, To:, Subject: etc.)
    3. Remove URLs
    4. Remove special characters and punctuation
    5. Lowercase
    6. Remove extra whitespace
    7. Tokenise, remove stop words, lemmatise
    8. Rejoin into clean string for TF-IDF
    """
    # Remove email headers
    text = re.sub(r'^(From|To|Subject|Cc|Bcc|Date|Message-ID|MIME|Content)[^\n]*\n',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove punctuation and special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Lowercase
    text = text.lower()

    # Tokenise
    tokens = text.split()

    # Remove stop words and short tokens
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]

    # Lemmatise
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def extract_features(texts, labels):
    """
    Full feature extraction pipeline:
    1. Text cleaning and lemmatisation
    2. Train/test split (before fitting vectorisers — no leakage)
    3. Word-level TF-IDF (unigrams + bigrams, max 6000 features)
    4. Character-level TF-IDF (3-5 char n-grams, max 4000 features)
    5. Feature concatenation (10000 dimensions)
    6. Chi-squared feature selection (top 7000 features)
    """
    # Step 1: Clean and lemmatise all texts
    print("Cleaning and lemmatising texts...")
    texts = [clean_text(str(t)) for t in texts]

    # Step 2: Train/test split BEFORE fitting vectorisers (prevents leakage)
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Step 3: Word-level TF-IDF — fitted on training data only
    word_vec = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    X_train_word = word_vec.fit_transform(texts_train)
    X_test_word = word_vec.transform(texts_test)

    # Step 4: Character-level TF-IDF — fitted on training data only
    # Captures obfuscation tactics (e.g. "fr33", "c1ick", "V1AGRA")
    char_vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=4000,
        sublinear_tf=True
    )
    X_train_char = char_vec.fit_transform(texts_train)
    X_test_char = char_vec.transform(texts_test)

    # Step 5: Concatenate word and character features (10000 dimensions)
    X_train = hstack([X_train_word, X_train_char])
    X_test = hstack([X_test_word, X_test_char])

    # Step 6: Chi-squared feature selection — fitted on training data only
    # Retains top 7000 features most statistically associated with class label
    selector = SelectKBest(chi2, k=7000)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    print(f"Feature matrix shape — Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, (word_vec, char_vec, selector)

def save_pipeline(word_vec, char_vec, selector, path="models/tfidf.pkl"):
    import joblib
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        'word_vec': word_vec,
        'char_vec': char_vec,
        'selector': selector
    }, path)
    print(f"Pipeline saved to {path}")