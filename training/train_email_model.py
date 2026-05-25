"""
Email Model Training Script
============================
Trains a Hybrid Ensemble: Random Forest + Naive Bayes + SVM + XGBoost
combined with Bidirectional LSTM (Keras) for spam/phishing classification.

Usage:
    python training/train_email_model.py
    python training/train_email_model.py --sample 50000   # fast mode
    python training/train_email_model.py --full           # use all data
"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS    = os.path.join(ROOT, "..", "datasets")
MODELS_DIR  = os.path.join(ROOT, "models", "saved")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── CLI Args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Email Phishing/Spam Detection Models")
parser.add_argument("--sample", type=int, default=50000,
                    help="Number of rows to sample (default 50000). Use 0 for all data.")
parser.add_argument("--full", action="store_true", help="Use full dataset (overrides --sample)")
parser.add_argument("--skip-lstm", action="store_true", help="Skip BiLSTM training (faster)")
args = parser.parse_args()

SAMPLE_SIZE = 0 if args.full else args.sample
TRAIN_LSTM  = not args.skip_lstm

print("=" * 65)
print("  AI Phishing & Spam Email Detection — Model Training")
print("=" * 65)

# ── 1. Load Datasets ──────────────────────────────────────────────────────────
print("\n📂 Loading datasets...")
dfs = []

# emails.csv (Enron-style: label, text)
emails_path = os.path.join(DATASETS, "emails.csv")
if os.path.exists(emails_path):
    try:
        df_e = pd.read_csv(emails_path, encoding="latin-1", low_memory=False)
        # Try to find label + text columns
        label_col = next((c for c in df_e.columns if "label" in c.lower() or "spam" in c.lower()), None)
        text_col  = next((c for c in df_e.columns if "text" in c.lower() or "body" in c.lower() or "message" in c.lower()), None)
        if label_col and text_col:
            df_e = df_e[[label_col, text_col]].dropna()
            df_e.columns = ["label", "text"]
            # Normalise label: 1 → Spam, 0 → Safe
            if df_e["label"].dtype != object:
                df_e["label"] = df_e["label"].map({1: "Spam", 0: "Safe"})
            dfs.append(df_e)
            print(f"  ✅ emails.csv loaded: {len(df_e):,} rows")
    except Exception as e:
        print(f"  ⚠️  emails.csv: {e}")

# emails1.csv
emails1_path = os.path.join(DATASETS, "emails1.csv")
if os.path.exists(emails1_path):
    try:
        df_e1 = pd.read_csv(emails1_path, encoding="latin-1", low_memory=False)
        label_col = next((c for c in df_e1.columns if "label" in c.lower() or "spam" in c.lower()), None)
        text_col  = next((c for c in df_e1.columns if "text" in c.lower() or "body" in c.lower() or "message" in c.lower()), None)
        if label_col and text_col:
            df_e1 = df_e1[[label_col, text_col]].dropna()
            df_e1.columns = ["label", "text"]
            if df_e1["label"].dtype != object:
                df_e1["label"] = df_e1["label"].map({1: "Spam", 0: "Safe"})
            dfs.append(df_e1)
            print(f"  ✅ emails1.csv loaded: {len(df_e1):,} rows")
    except Exception as e:
        print(f"  ⚠️  emails1.csv: {e}")

# Phishing_Email.csv
phish_path = os.path.join(DATASETS, "Phishing_Email.csv")
if os.path.exists(phish_path):
    try:
        chunk_size = 100_000
        chunks = []
        max_phish = 150_000 if not args.full else None
        total_read = 0
        for chunk in pd.read_csv(phish_path, encoding="latin-1",
                                  low_memory=False, chunksize=chunk_size):
            # Find label/text columns
            label_col = next((c for c in chunk.columns
                              if "label" in c.lower() or "type" in c.lower()
                              or "class" in c.lower() or "spam" in c.lower()), None)
            text_col  = next((c for c in chunk.columns
                              if "text" in c.lower() or "body" in c.lower()
                              or "email" in c.lower() or "content" in c.lower()), None)
            if not label_col or not text_col:
                # Try first two columns
                label_col, text_col = chunk.columns[0], chunk.columns[1]

            sub = chunk[[label_col, text_col]].dropna()
            sub.columns = ["label", "text"]
            # Normalise label
            sub["label"] = sub["label"].astype(str).str.strip()
            mapping = {
                "1": "Phishing", "phishing": "Phishing", "spam": "Spam",
                "0": "Safe", "safe": "Safe", "ham": "Safe", "legitimate": "Safe",
            }
            sub["label"] = sub["label"].str.lower().map(mapping).fillna("Phishing")
            chunks.append(sub)
            total_read += len(sub)
            if max_phish and total_read >= max_phish:
                break

        if chunks:
            df_p = pd.concat(chunks, ignore_index=True)
            dfs.append(df_p)
            print(f"  ✅ Phishing_Email.csv loaded: {len(df_p):,} rows")
    except Exception as e:
        print(f"  ⚠️  Phishing_Email.csv: {e}")

def generate_synthetic_emails(n=1000):
    import random
    safe_texts = [
        "Hi Team, please find the attached project report. Let me know if you have any questions.",
        "Hello, are we still meeting today at 3 PM? Let me know.",
        "Hi, thanks for the quick response. I will review the documents and get back to you.",
        "Good morning, here is the agenda for today's standup meeting.",
        "Dear customer, your order has been shipped and will arrive shortly."
    ]
    spam_texts = [
        "CONGRATULATIONS! You have won a free lottery prize of $1,000,000! Click here now!",
        "Get cheap replica watches and luxury bags at 90% discount! Limited offer!",
        "Special pharmacy deal: buy viagra and cialis online cheap without prescription!",
        "Earn $5000 a day working from home! No experience required! Sign up today!",
        "Invest in bitcoin now and double your money in 24 hours guaranteed!"
    ]
    phish_texts = [
        "Urgent: your bank account has been suspended. Please login to verify your identity immediately.",
        "Verify your PayPal account secure sign in link. Update your billing details now.",
        "Microsoft Security Alert: Unauthorized sign-in detected. Click here to secure your account.",
        "Attention: your email box is full. Please verify your credentials to avoid account closure.",
        "Document shared with you on OneDrive. Sign in with your email account to view."
    ]
    
    rows = []
    for _ in range(n):
        label = random.choice(["Safe", "Spam", "Phishing"])
        if label == "Safe":
            text = random.choice(safe_texts)
        elif label == "Spam":
            text = random.choice(spam_texts)
        else:
            text = random.choice(phish_texts)
        rows.append({"label": label, "text": text})
    return pd.DataFrame(rows)

has_label_and_text = False
if dfs:
    try:
        temp_df = pd.concat(dfs, ignore_index=True)
        if "label" in temp_df.columns and "text" in temp_df.columns:
            has_label_and_text = True
    except Exception:
        pass

if not dfs or not has_label_and_text:
    print("ℹ️ No valid email datasets found or failed to parse. Generating synthetic training data...")
    df = generate_synthetic_emails(2000)
else:
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["label", "text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 10]

# Sample if requested
if SAMPLE_SIZE > 0 and len(df) > SAMPLE_SIZE:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // df["label"].nunique()),
                            random_state=42)
    ).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n📊 Sampled to {len(df):,} rows")

print(f"\n📊 Dataset summary:")
print(df["label"].value_counts().to_string())
print(f"   Total: {len(df):,}")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
print("\n🔧 Preprocessing text...")
sys.path.insert(0, ROOT)
from utils.preprocessor import preprocess

df["processed"] = df["text"].apply(preprocess)
df = df[df["processed"].str.len() > 3]

# ── 3. Encode Labels ──────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df["label"])
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
print(f"   Classes: {list(le.classes_)}")

# ── 4. TF-IDF Vectorization ───────────────────────────────────────────────────
print("\n🔤 Building TF-IDF features...")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=15_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
)
X = vectorizer.fit_transform(df["processed"])
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
print(f"   Feature matrix: {X.shape}")

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ── 6. Train Individual ML Models ────────────────────────────────────────────
print("\n🤖 Training ML models...")

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Naive Bayes
print("   [1/4] Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test))
print(f"         Accuracy: {nb_acc:.4f}")

# SVM (LinearSVC — needs calibration for probabilities)
print("   [2/4] SVM (LinearSVC + Calibration)...")
svc = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3)
svc.fit(X_train, y_train)
svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(f"         Accuracy: {svc_acc:.4f}")

# Random Forest
print("   [3/4] Random Forest...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"         Accuracy: {rf_acc:.4f}")

# XGBoost
print("   [4/4] XGBoost...")
n_classes = len(le.classes_)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss" if n_classes > 2 else "logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
print(f"         Accuracy: {xgb_acc:.4f}")

# Save individual models
joblib.dump(nb,  os.path.join(MODELS_DIR, "naive_bayes.pkl"))
joblib.dump(svc, os.path.join(MODELS_DIR, "svm.pkl"))
joblib.dump(rf,  os.path.join(MODELS_DIR, "random_forest.pkl"))
joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))

# ── 7. Ensemble (Soft Voting) ─────────────────────────────────────────────────
print("\n🎯 Building Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[("nb", nb), ("svm", svc), ("rf", rf), ("xgb", xgb)],
    voting="soft",
    weights=[1, 2, 2, 3],
)
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"   Ensemble Accuracy: {ens_acc:.4f}")
print(classification_report(y_test, ensemble.predict(X_test), target_names=le.classes_))
joblib.dump(ensemble, os.path.join(MODELS_DIR, "email_ensemble.pkl"))

# ── 8. BiLSTM (Keras) ────────────────────────────────────────────────────────
if TRAIN_LSTM:
    print("\n🧠 Training BiLSTM Deep Learning Model...")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
        )
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        MAX_WORDS = 20_000
        MAX_LEN   = 300
        EMBED_DIM = 128

        tokenizer_keras = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer_keras.fit_on_texts(df["processed"].tolist())
        joblib.dump(tokenizer_keras, os.path.join(MODELS_DIR, "keras_tokenizer.pkl"))

        seqs   = tokenizer_keras.texts_to_sequences(df["processed"].tolist())
        padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

        X_tr, X_te, y_tr, y_te = train_test_split(
            padded, y, test_size=0.2, random_state=42, stratify=y
        )

        model = Sequential([
            Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)),
            Bidirectional(LSTM(64, dropout=0.3)),
            Dense(64, activation="relu"),
            Dropout(0.4),
            Dense(n_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
            ReduceLROnPlateau(patience=2, factor=0.5),
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=10,
            batch_size=128,
            callbacks=callbacks,
        )

        lstm_acc = model.evaluate(X_te, y_te, verbose=0)[1]
        print(f"   BiLSTM Test Accuracy: {lstm_acc:.4f}")
        model.save(os.path.join(MODELS_DIR, "bilstm_model.h5"))
        print("   ✅ BiLSTM saved")

    except Exception as e:
        print(f"   ⚠️  LSTM training failed: {e}")
        TRAIN_LSTM = False

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("✅ Training Complete!")
print(f"   Naive Bayes  : {nb_acc:.4f}")
print(f"   SVM          : {svc_acc:.4f}")
print(f"   Random Forest: {rf_acc:.4f}")
print(f"   XGBoost      : {xgb_acc:.4f}")
print(f"   Ensemble     : {ens_acc:.4f}")
print(f"\n   Models saved to: {MODELS_DIR}")
print("=" * 65)
