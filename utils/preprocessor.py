"""
NLP Preprocessor — text cleaning, tokenization, lemmatization, URL extraction.
"""

import re
import string
from typing import List, Tuple
from urllib.parse import urlparse

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only once)
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Suspicious keywords for phishing/spam detection
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "account", "suspended", "click here", "free", "winner",
    "prize", "lottery", "credit card", "bank", "password", "login", "confirm",
    "limited time", "act now", "congratulations", "claim", "offer", "discount",
    "risk free", "guarantee", "money back", "million", "billion", "inheritance",
    "prince", "nigeria", "wire transfer", "paypal", "bitcoin", "crypto",
    "unsubscribe", "opt out", "remove", "dear customer", "dear user",
    "your account", "verify your", "update your", "confirm your",
    "click the link", "click below", "follow the link", "suspicious activity",
    "unauthorized", "unusual activity", "security alert", "immediately",
    "expire", "expires", "deadline", "24 hours", "48 hours",
]

URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def clean_text(text: str) -> str:
    """Remove HTML tags, URLs, special chars and normalise whitespace."""
    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(URL_PATTERN, " URL ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep spaces)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> List[str]:
    """Tokenize, remove stopwords, and lemmatize."""
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in _stop_words and len(t) > 2
    ]
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline: clean → tokenize → lemmatize → join."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from raw text."""
    return list(set(URL_PATTERN.findall(text)))


def find_suspicious_words(text: str) -> List[str]:
    """Return suspicious keywords found in the text."""
    lower_text = text.lower()
    found = []
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in lower_text:
            found.append(kw)
    return list(set(found))


# ── URL Feature Extraction ────────────────────────────────────────────────────
def extract_url_features(url: str) -> dict:
    """Extract numerical features from a URL for phishing detection."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        full = url

        features = {
            "url_length": len(url),
            "hostname_length": len(hostname),
            "path_length": len(path),
            "num_dots_hostname": hostname.count("."),
            "num_hyphens": url.count("-"),
            "num_at_sign": url.count("@"),
            "num_double_slash": url.count("//"),
            "num_slash": url.count("/"),
            "num_question_mark": url.count("?"),
            "num_equal": url.count("="),
            "num_ampersand": url.count("&"),
            "has_https": int(parsed.scheme == "https"),
            "has_ip_address": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname))),
            "has_port": int(parsed.port is not None),
            "num_subdomains": len(hostname.split(".")) - 2 if hostname else 0,
            "has_www": int(hostname.startswith("www.")),
            "num_digits_in_hostname": sum(c.isdigit() for c in hostname),
            "has_suspicious_words": int(
                any(w in url.lower() for w in [
                    "login", "signin", "verify", "account", "update",
                    "secure", "banking", "paypal", "confirm", "password",
                ])
            ),
            "tld_in_path": int(
                any(tld in path.lower() for tld in [".com", ".net", ".org", ".gov"])
            ),
            "has_encoded_chars": int("%" in url),
        }
        return features
    except Exception:
        return {k: 0 for k in [
            "url_length", "hostname_length", "path_length",
            "num_dots_hostname", "num_hyphens", "num_at_sign",
            "num_double_slash", "num_slash", "num_question_mark",
            "num_equal", "num_ampersand", "has_https", "has_ip_address",
            "has_port", "num_subdomains", "has_www",
            "num_digits_in_hostname", "has_suspicious_words",
            "tld_in_path", "has_encoded_chars",
        ]}
