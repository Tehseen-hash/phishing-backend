"""
Email Scanner API — Full email analysis using Hybrid Ensemble AI.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services.prediction_service import get_predictor

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────
class EmailScanRequest(BaseModel):
    subject: Optional[str] = ""
    body: str
    sender: Optional[str] = ""
    attachments: Optional[List[str]] = []


class EmailScanResponse(BaseModel):
    label: str                    # "Safe" | "Spam" | "Phishing"
    spam_probability: float
    phishing_probability: float
    safe_probability: float
    confidence_score: float
    threat_level: str             # "Low" | "Medium" | "High" | "Critical"
    suspicious_words: List[str]
    extracted_urls: List[str]
    url_risk_scores: dict
    model_votes: dict
    analysis_summary: str


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/email", response_model=EmailScanResponse)
async def scan_email(req: EmailScanRequest):
    """
    Analyze an email using the Hybrid Ensemble AI model.
    Combines ML (RF, SVM, NB, XGBoost) + DL (BiLSTM) predictions.
    """
    if not req.body or len(req.body.strip()) < 5:
        raise HTTPException(status_code=422, detail="Email body is too short to analyze")

    try:
        predictor = get_predictor()
        result = predictor.predict_email(
            subject=req.subject or "",
            body=req.body,
            sender=req.sender or "",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
