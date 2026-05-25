"""
History API — Scan history management (stored in local SQLite database).
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List
from utils.security import get_current_user
from utils.db import get_connection

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────
class HistoryEntry(BaseModel):
    scan_type: str          # "email" | "url"
    input_text: str
    result_label: str
    confidence: float
    threat_level: str
    user_id: Optional[str] = None


class HistoryResponse(BaseModel):
    id: str
    scan_type: str
    input_text: str
    result_label: str
    confidence: float
    threat_level: str
    timestamp: str
    user_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Return aggregate stats for the current user's scan history."""
    conn = get_connection()
    cursor = conn.cursor()
    user_id = current_user["sub"]
    try:
        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,))
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND result_label = 'Spam'", (user_id,))
        spam = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND result_label = 'Phishing'", (user_id,))
        phishing = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND result_label = 'Safe'", (user_id,))
        safe = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND scan_type = 'email'", (user_id,))
        email_scans = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ? AND scan_type = 'url'", (user_id,))
        url_scans = cursor.fetchone()[0]
    finally:
        conn.close()

    return {
        "total_scans": total,
        "spam_detected": spam,
        "phishing_detected": phishing,
        "safe_detected": safe,
        "email_scans": email_scans,
        "url_scans": url_scans,
    }


@router.get("/", response_model=List[HistoryResponse])
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    scan_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Return paginated scan history for the current user."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM history WHERE user_id = ?"
    params: list = [current_user["sub"]]

    if scan_type:
        query += " AND scan_type = ?"
        params.append(scan_type)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    try:
        cursor.execute(query, tuple(params))
        records = cursor.fetchall()
    finally:
        conn.close()

    return [dict(r) for r in records]


@router.post("/", response_model=HistoryResponse, status_code=201)
async def add_history(
    entry: HistoryEntry,
    current_user: dict = Depends(get_current_user),
):
    """Add a new scan result to the user's history."""
    conn = get_connection()
    cursor = conn.cursor()

    entry_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    try:
        # Insert history entry into SQLite
        cursor.execute(
            "INSERT INTO history (id, user_id, scan_type, input_text, result_label, confidence, threat_level, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entry_id, current_user["sub"], entry.scan_type, entry.input_text[:500],
             entry.result_label, entry.confidence, entry.threat_level, timestamp)
        )

        # Increment scan count of user
        cursor.execute(
            "UPDATE users SET scan_count = scan_count + 1 WHERE id = ?",
            (current_user["sub"],)
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()

    return {
        "id": entry_id,
        "scan_type": entry.scan_type,
        "input_text": entry.input_text[:500],
        "result_label": entry.result_label,
        "confidence": entry.confidence,
        "threat_level": entry.threat_level,
        "user_id": current_user["sub"],
        "timestamp": timestamp,
    }


# NOTE: DELETE "/" must come BEFORE DELETE "/{entry_id}" to avoid route shadowing
@router.delete("/", status_code=204)
async def clear_history(current_user: dict = Depends(get_current_user)):
    """Delete ALL history entries for the current user."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM history WHERE user_id = ?", (current_user["sub"],))
        conn.commit()
    finally:
        conn.close()


@router.delete("/{entry_id}", status_code=204)
async def delete_history(
    entry_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a specific history entry by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "DELETE FROM history WHERE id = ? AND user_id = ?",
            (entry_id, current_user["sub"])
        )
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="History entry not found")
        conn.commit()
    finally:
        conn.close()
