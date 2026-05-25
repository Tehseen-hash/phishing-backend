"""
Prediction Service — loads trained models and serves predictions.
Falls back to a rule-based heuristic system if models are not yet trained.

Uses a module-level singleton so PredictionService is instantiated ONCE
across all API routers (avoids duplicate model loads and double warnings).
"""

import os
import re
from typing import Optional, Tuple

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import joblib
    _JOBLIB = True
except ImportError:
    joblib = None
    _JOBLIB = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    np = None
    _NUMPY = False

from utils.preprocessor import (
    preprocess, extract_urls, find_suspicious_words, extract_url_features
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")


class PredictionService:
    """
    Hybrid Ensemble prediction service.
    Loads: TF-IDF vectorizer, ML ensemble, BiLSTM (Keras), URL XGBoost.
    Falls back to heuristics when models are not yet trained.
    """

    def __init__(self):
        self.email_vectorizer = None
        self.email_ensemble = None
        self.url_model = None
        self.keras_model = None
        self.tokenizer = None
        self._load_models()

    def _load_models(self):
        if not _JOBLIB:
            print("WARNING: joblib not available — using heuristic fallback only")
            return

        try:
            vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
            ens_path = os.path.join(MODELS_DIR, "email_ensemble.pkl")
            url_path = os.path.join(MODELS_DIR, "url_model.pkl")

            if os.path.exists(vec_path):
                self.email_vectorizer = joblib.load(vec_path)
                print("SUCCESS: TF-IDF vectorizer loaded")

            if os.path.exists(ens_path):
                self.email_ensemble = joblib.load(ens_path)
                print("SUCCESS: Email ensemble model loaded")
            else:
                print("INFO: email_ensemble.pkl not found — run training/train_email_model.py to train")

            if os.path.exists(url_path):
                self.url_model = joblib.load(url_path)
                print("SUCCESS: URL phishing model loaded")
            else:
                print("INFO: url_model.pkl not found — using heuristic URL analysis")

            # Try loading Keras BiLSTM (tensorflow optional)
            try:
                import tensorflow as tf  # noqa: F401
                tok_path = os.path.join(MODELS_DIR, "keras_tokenizer.pkl")
                lstm_path = os.path.join(MODELS_DIR, "bilstm_model.h5")
                if os.path.exists(lstm_path) and os.path.exists(tok_path):
                    self.keras_model = tf.keras.models.load_model(lstm_path)
                    self.tokenizer = joblib.load(tok_path)
                    print("SUCCESS: BiLSTM model loaded")
                else:
                    print("INFO: BiLSTM model not found — install tensorflow and run training to enable")
            except ModuleNotFoundError:
                print("INFO: tensorflow not installed — BiLSTM skipped (heuristic fallback active)")
            except Exception as e:
                print(f"WARNING: Keras model not loaded: {e}")

        except Exception as e:
            print(f"WARNING: Model loading error: {e} — using heuristic fallback")

    # ──────────────────────────────────────────────────────────────────────────
    # EMAIL PREDICTION
    # ──────────────────────────────────────────────────────────────────────────
    def predict_email(self, subject: str, body: str, sender: str) -> dict:
        full_text = f"{subject} {body}"
        urls = extract_urls(full_text)
        suspicious_words = find_suspicious_words(full_text)

        # URL risk scores
        url_risk_scores = {}
        for url in urls[:10]:
            try:
                url_result = self.predict_url(url)
                url_risk_scores[url] = url_result.get("risk_score", 0.0)
            except Exception:
                url_risk_scores[url] = 0.0

        # ── ML Ensemble prediction ────────────────────────────────────────────
        if self.email_vectorizer and self.email_ensemble:
            spam_prob, phish_prob, safe_prob, votes = self._ml_predict_email(
                full_text, subject, sender, suspicious_words, url_risk_scores
            )
        else:
            spam_prob, phish_prob, safe_prob, votes = self._heuristic_email(
                full_text, subject, sender, suspicious_words, url_risk_scores
            )

        # ── BiLSTM blending ───────────────────────────────────────────────────
        if self.keras_model and self.tokenizer:
            try:
                lstm_spam, lstm_phish = self._lstm_predict(full_text)
                # Blend: 60% ensemble + 40% LSTM
                spam_prob  = 0.6 * spam_prob  + 0.4 * lstm_spam
                phish_prob = 0.6 * phish_prob + 0.4 * lstm_phish
                safe_prob  = max(0.0, 1.0 - spam_prob - phish_prob)
                votes["BiLSTM"] = "Spam" if lstm_spam > 0.5 else ("Phishing" if lstm_phish > 0.5 else "Safe")
            except Exception as e:
                print(f"WARNING: LSTM prediction error: {e}")

        # ── Final label ───────────────────────────────────────────────────────
        probs = {"Safe": safe_prob, "Spam": spam_prob, "Phishing": phish_prob}
        label = max(probs, key=probs.get)
        confidence = round(probs[label] * 100, 1)
        threat_level = _threat_level(max(spam_prob, phish_prob))

        return {
            "label": label,
            "spam_probability": round(spam_prob * 100, 2),
            "phishing_probability": round(phish_prob * 100, 2),
            "safe_probability": round(safe_prob * 100, 2),
            "confidence_score": confidence,
            "threat_level": threat_level,
            "suspicious_words": suspicious_words[:20],
            "extracted_urls": urls[:10],
            "url_risk_scores": {k: round(v * 100, 1) for k, v in url_risk_scores.items()},
            "model_votes": votes,
            "analysis_summary": _generate_summary(label, confidence, suspicious_words, urls),
        }

    def _ml_predict_email(
        self, text: str, subject: str, sender: str,
        suspicious_words: list, url_risks: dict
    ) -> Tuple[float, float, float, dict]:
        try:
            preprocessed = preprocess(text)
            vec = self.email_vectorizer.transform([preprocessed])

            proba = self.email_ensemble.predict_proba(vec)[0]
            n_classes = len(proba)

            if n_classes == 3:
                safe_prob, spam_prob, phish_prob = float(proba[0]), float(proba[1]), float(proba[2])
            else:
                spam_prob = float(proba[1]) if n_classes > 1 else float(proba[0])
                safe_prob = float(proba[0])
                phish_prob = 0.0

            # Apply URL risk boost
            max_url_risk = max(url_risks.values(), default=0.0)
            phish_boost = max_url_risk * 0.3
            phish_prob = min(1.0, phish_prob + phish_boost)
            safe_prob = max(0.0, safe_prob - phish_boost)

            votes = {
                "EnsembleML": "Spam" if spam_prob > 0.5 else ("Phishing" if phish_prob > 0.5 else "Safe")
            }
            return spam_prob, phish_prob, safe_prob, votes
        except Exception as e:
            print(f"WARNING: ML prediction failed ({e}), falling back to heuristic")
            return self._heuristic_email(text, subject, sender, suspicious_words, url_risks)

    def _lstm_predict(self, text: str) -> Tuple[float, float]:
        from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
        MAX_LEN = 300
        seq = self.tokenizer.texts_to_sequences([preprocess(text)])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        preds = self.keras_model.predict(padded, verbose=0)[0]
        if len(preds) >= 2:
            return float(preds[0]), float(preds[1])
        return float(preds[0]), 0.0

    def _heuristic_email(
        self, text: str, subject: str, sender: str,
        suspicious_words: list, url_risks: dict
    ) -> Tuple[float, float, float, dict]:
        """Rule-based fallback when models are not trained yet."""
        lower = text.lower()
        score = 0.0

        # Suspicious word count
        score += min(len(suspicious_words) * 0.05, 0.5)

        # Urgency patterns
        if re.search(r"\b(urgent|immediately|act now|limited time|expire)\b", lower):
            score += 0.2

        # Sender checks
        if sender:
            sender_lower = sender.lower()
            if re.search(r"\d{4,}", sender_lower):
                score += 0.15
            if not re.search(r"@[a-zA-Z]+\.[a-zA-Z]{2,4}$", sender_lower):
                score += 0.1

        # URL risks
        max_url_risk = max(url_risks.values(), default=0.0)
        phish_component = max_url_risk * 0.5

        # Is this more phishing or spam?
        is_phish = phish_component > 0.3 or bool(re.search(
            r"(verify|confirm|login|signin|account|bank|paypal)", lower
        ))

        if is_phish:
            phish_prob = min(0.95, score + phish_component)
            spam_prob = score * 0.3
        else:
            spam_prob = min(0.95, score)
            phish_prob = phish_component

        safe_prob = max(0.0, 1.0 - spam_prob - phish_prob)
        votes = {"Heuristic": "Spam" if spam_prob > 0.5 else ("Phishing" if phish_prob > 0.5 else "Safe")}
        return spam_prob, phish_prob, safe_prob, votes

    # ──────────────────────────────────────────────────────────────────────────
    # URL PREDICTION
    # ──────────────────────────────────────────────────────────────────────────
    def predict_url(self, url: str) -> dict:
        features = extract_url_features(url)

        if self.url_model and _NUMPY:
            try:
                feat_array = np.array([list(features.values())])
                proba = self.url_model.predict_proba(feat_array)[0]
                phish_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception as e:
                print(f"WARNING: URL model prediction failed ({e}), using heuristic")
                phish_prob = self._heuristic_url(url, features)
        else:
            phish_prob = self._heuristic_url(url, features)

        safe_prob = 1.0 - phish_prob
        is_phishing = phish_prob >= 0.5
        threat_level = _threat_level(phish_prob)

        return {
            "url": url,
            "is_phishing": is_phishing,
            "risk_score": round(phish_prob, 4),
            "phishing_probability": round(phish_prob * 100, 2),
            "safe_probability": round(safe_prob * 100, 2),
            "threat_level": threat_level,
            "features": features,
            "verdict": f"{'⚠️ Phishing' if is_phishing else '✅ Safe'} URL — {threat_level} risk",
        }

    def _heuristic_url(self, url: str, features: dict) -> float:
        score = 0.0
        if features.get("has_ip_address"):
            score += 0.4
        if features.get("num_dots_hostname", 0) > 3:
            score += 0.2
        if not features.get("has_https"):
            score += 0.15
        if features.get("num_hyphens", 0) > 3:
            score += 0.1
        if features.get("url_length", 0) > 100:
            score += 0.1
        if features.get("has_suspicious_words"):
            score += 0.25
        if features.get("has_encoded_chars"):
            score += 0.1
        if features.get("num_at_sign", 0) > 0:
            score += 0.3
        return min(score, 0.99)


# ── Module-level singleton (prevents double instantiation across routers) ─────
_predictor_instance: Optional[PredictionService] = None


def get_predictor() -> PredictionService:
    """Return the shared singleton PredictionService instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PredictionService()
    return _predictor_instance


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _threat_level(prob: float) -> str:
    if prob < 0.2:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.8:
        return "High"
    else:
        return "Critical"


def _generate_summary(label: str, confidence: float, words: list, urls: list) -> str:
    parts = [f"This email is classified as {label} with {confidence}% confidence."]
    if words:
        parts.append(f"Found {len(words)} suspicious keyword(s): {', '.join(words[:5])}.")
    if urls:
        parts.append(f"Detected {len(urls)} URL(s) for further analysis.")
    if label == "Safe":
        parts.append("No significant threats detected.")
    elif label == "Spam":
        parts.append("This appears to be unsolicited bulk email. Exercise caution.")
    else:
        parts.append("⚠️ This email shows signs of a phishing attempt. Do NOT click links or share credentials.")
    return " ".join(parts)
