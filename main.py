"""
AI-Powered Phishing & Spam Email Detection System
FastAPI Backend - Main Application Entry Point
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

from api.auth import router as auth_router
from api.email_scanner import router as email_router
from api.url_scanner import router as url_router
from api.history import router as history_router

load_dotenv()

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan: initialise shared resources once on startup ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the singleton predictor at startup (not lazily per-request)
    from services.prediction_service import get_predictor
    get_predictor()
    yield
    # (shutdown logic goes here if needed)


# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Phishing & Spam Detection API",
    description=(
        "Hybrid Ensemble + Deep Learning system for detecting "
        "spam emails, phishing emails, and malicious URLs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router,    prefix="/auth",    tags=["Authentication"])
app.include_router(email_router,   prefix="/scan",    tags=["Email Scanner"])
app.include_router(url_router,     prefix="/scan",    tags=["URL Scanner"])
app.include_router(history_router, prefix="/history", tags=["History"])


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "app": "AI Phishing & Spam Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
