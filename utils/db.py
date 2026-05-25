"""
SQLite Database utility for FastAPI Backend.
Handles user authentication and scan history storage.
"""

import os
import sqlite3
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "phishguard.db")


def get_connection():
    """Establish a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    # Enable foreign key cascade constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as dictionaries
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite tables for users and history."""
    conn = get_connection()
    cursor = conn.cursor()

    # Create Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            scan_count INTEGER DEFAULT 0
        )
    """)

    # Create History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            scan_type TEXT NOT NULL,
            input_text TEXT NOT NULL,
            result_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            threat_level TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    print("Backend SQLite Database initialized successfully")


# Auto-initialize database on import
init_db()
