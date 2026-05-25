"""
Authentication API — SQLite-based user registration and login.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, field_validator
from utils.security import hash_password, verify_password, create_access_token, get_current_user
from utils.db import get_connection

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    conn = get_connection()
    cursor = conn.cursor()

    # Check duplicate email
    cursor.execute("SELECT id FROM users WHERE email = ?", (req.email.lower().strip(),))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    password_hash = hash_password(req.password)
    created_at = datetime.utcnow().isoformat()

    # Insert new user
    cursor.execute(
        "INSERT INTO users (id, username, email, password_hash, created_at, scan_count) VALUES (?, ?, ?, ?, ?, 0)",
        (user_id, req.username.strip(), req.email.lower().strip(), password_hash, created_at)
    )
    conn.commit()
    conn.close()

    token = create_access_token({"sub": user_id, "email": req.email.lower().strip()})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": req.username.strip(), "email": req.email.lower().strip()},
    }


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    conn = get_connection()
    cursor = conn.cursor()

    # Find user
    cursor.execute("SELECT * FROM users WHERE email = ?", (req.email.lower().strip(),))
    user = cursor.fetchone()
    conn.close()

    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user["id"], "email": user["email"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "scan_count": user["scan_count"],
        },
    }


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    cursor = conn.cursor()

    # Fetch user data
    cursor.execute("SELECT * FROM users WHERE id = ?", (current_user["sub"],))
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "scan_count": user["scan_count"],
        "created_at": user["created_at"],
    }


class UpdatePasswordRequest(BaseModel):
    old_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


@router.post("/update-password")
async def update_password(req: UpdatePasswordRequest, current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    cursor = conn.cursor()

    # Get user
    cursor.execute("SELECT * FROM users WHERE id = ?", (current_user["sub"],))
    user = cursor.fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    # Verify old password
    if not verify_password(req.old_password, user["password_hash"]):
        conn.close()
        raise HTTPException(status_code=400, detail="Incorrect old password")

    # Hash and save new password
    new_hash = hash_password(req.new_password)
    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, current_user["sub"]))
    conn.commit()
    conn.close()

    return {"detail": "Password updated successfully"}


@router.delete("/delete-account")
async def delete_account(current_user: dict = Depends(get_current_user)):
    conn = get_connection()
    cursor = conn.cursor()

    # Get user
    cursor.execute("SELECT * FROM users WHERE id = ?", (current_user["sub"],))
    user = cursor.fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    # Delete user (foreign key cascade deletes their history entries)
    cursor.execute("DELETE FROM users WHERE id = ?", (current_user["sub"],))
    conn.commit()
    conn.close()

    return {"detail": "Account deleted successfully"}

