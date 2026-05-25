"""
URL Scanner API — Phishing URL risk analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.prediction_service import get_predictor

router = APIRouter()


class URLScanRequest(BaseModel):
    url: str


class URLScanResponse(BaseModel):
    url: str
    is_phishing: bool
    risk_score: float           # 0.0 – 1.0
    phishing_probability: float
    safe_probability: float
    threat_level: str           # "Low" | "Medium" | "High" | "Critical"
    features: dict
    verdict: str


class BulkURLScanRequest(BaseModel):
    urls: List[str]


@router.post("/url", response_model=URLScanResponse)
async def scan_url(req: URLScanRequest):
    """Analyze a single URL for phishing indicators."""
    if not req.url or len(req.url.strip()) < 4:
        raise HTTPException(status_code=422, detail="Invalid URL provided")

    try:
        predictor = get_predictor()
        result = predictor.predict_url(req.url.strip())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis error: {str(e)}")


@router.post("/urls/bulk")
async def scan_urls_bulk(req: BulkURLScanRequest):
    """Analyze multiple URLs at once (max 20)."""
    if len(req.urls) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 URLs per request")

    predictor = get_predictor()
    results = []
    for url in req.urls:
        try:
            result = predictor.predict_url(url.strip())
            results.append(result)
        except Exception:
            results.append({"url": url, "error": "Analysis failed"})

    return {"results": results, "total": len(results)}
