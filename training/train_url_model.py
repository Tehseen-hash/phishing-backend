"""
URL Phishing Detection Model Training
======================================
Extracts features from URLs and trains an XGBoost classifier.

Usage:
    python training/train_url_model.py
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS   = os.path.join(ROOT, "..", "datasets")
MODELS_DIR = os.path.join(ROOT, "models", "saved")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from utils.preprocessor import extract_url_features

print("=" * 55)
print("  URL Phishing Detection — Model Training")
print("=" * 55)

# ── Generate synthetic URL dataset (used when no real URL dataset exists) ────
def generate_synthetic_urls(n=20000):
    import random, string

    def rand_str(length):
        return "".join(random.choices(string.ascii_lowercase, k=length))

    safe_domains = [
        "google.com", "facebook.com", "microsoft.com", "apple.com",
        "amazon.com", "github.com", "wikipedia.org", "stackoverflow.com",
        "youtube.com", "twitter.com", "linkedin.com", "reddit.com",
    ]
    phish_patterns = [
        "http://{ip}/login",
        "http://{domain}-secure-{rand}.com/verify/account",
        "http://{rand}.{rand}.xyz/paypal-login",
        "http://192.168.{r}.{r}/banking/signin",
        "http://{rand}{rand}{rand}.ru/update-password",
        "http://{domain}.{rand}-confirm.info/phish",
        "https://secure-{rand}.verify-{rand}.net/login?token={rand}",
    ]

    rows = []
    # Safe URLs
    for _ in range(n // 2):
        domain = random.choice(safe_domains)
        path = "/" + "/".join(rand_str(random.randint(3, 8)) for _ in range(random.randint(0, 3)))
        url = f"https://www.{domain}{path}"
        rows.append({"url": url, "label": 0})

    # Phishing URLs
    for _ in range(n // 2):
        pattern = random.choice(phish_patterns)
        r = random.randint(1, 254)
        url = pattern.format(
            ip=f"{r}.{r}.{r}.{r}",
            domain=random.choice(["paypal", "bank", "amazon", "google", "microsoft"]),
            rand=rand_str(random.randint(5, 12)),
            r=r,
        )
        rows.append({"url": url, "label": 1})

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


# Try loading real URL dataset or generate synthetic
url_dataset_path = os.path.join(DATASETS, "phishing_urls.csv")
if os.path.exists(url_dataset_path):
    df = pd.read_csv(url_dataset_path, encoding="latin-1", low_memory=False)
    label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)
    url_col   = next((c for c in df.columns if "url" in c.lower() or "address" in c.lower()), None)
    if label_col and url_col:
        df = df[[url_col, label_col]].dropna()
        df.columns = ["url", "label"]
        df["label"] = (df["label"].astype(str).str.lower().isin(["1", "phishing"])).astype(int)
        print(f"✅ Loaded real URL dataset: {len(df):,} rows")
    else:
        print("⚠️  URL dataset columns not found. Generating synthetic...")
        df = generate_synthetic_urls(40000)
else:
    print("ℹ️  No URL dataset found. Generating synthetic training data...")
    df = generate_synthetic_urls(40000)

print(f"   Phishing: {df['label'].sum():,} | Safe: {(df['label'] == 0).sum():,}")

# ── Feature Extraction ────────────────────────────────────────────────────────
print("\n🔧 Extracting URL features...")
from tqdm import tqdm
tqdm.pandas()
features_df = df["url"].progress_apply(lambda u: pd.Series(extract_url_features(u)))
X = features_df.values.astype(float)
y = df["label"].values
print(f"   Feature matrix: {X.shape}")

# ── Train / Test Split ────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("\n🤖 Training XGBoost URL model...")
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

preds = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

acc   = accuracy_score(y_test, preds)
auc   = roc_auc_score(y_test, proba)

print(f"\n   Accuracy : {acc:.4f}")
print(f"   ROC-AUC  : {auc:.4f}")
print(classification_report(y_test, preds, target_names=["Safe", "Phishing"]))

# Save model and feature names
joblib.dump(model, os.path.join(MODELS_DIR, "url_model.pkl"))
joblib.dump(list(features_df.columns), os.path.join(MODELS_DIR, "url_feature_names.pkl"))

print("\n✅ URL model saved!")
print(f"   Saved to: {MODELS_DIR}")
