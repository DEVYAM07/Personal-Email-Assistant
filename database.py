"""Database helpers for SQLite.

Creates `users` table for per-user OAuth credentials and provides
helpers for managing the users and emails tables.
"""

import os
import sqlite3

DB_PATH = os.getenv("DB_PATH") or os.getenv("SQLITE_PATH") or os.path.join(os.path.dirname(__file__), "emails.db")


def get_db_connection():
    """Return a new SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    """Initialize database tables (users + emails) if not present."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # users table for dynamic OAuth credentials
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            refresh_token TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # emails table used by sync / vector_store
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            subject TEXT,
            from_addr TEXT,
            date TEXT,
            body TEXT,
            snippet TEXT,
            fetched_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def init_users_table():
    """Ensure users table exists (alias for backwards compatibility)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            refresh_token TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_user(email: str, refresh_token: str):
    """Insert or update a user refresh token."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    init_users_table()
    cursor.execute(
        """
        INSERT INTO users (email, refresh_token, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(email) DO UPDATE SET
            refresh_token=excluded.refresh_token,
            updated_at=CURRENT_TIMESTAMP
        """,
        (email, refresh_token),
    )
    conn.commit()
    conn.close()


def get_refresh_token(email: str):
    """Retrieve refresh_token for email or None if not found."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT refresh_token FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def is_authenticated(email: str) -> bool:
    """Check if user has a stored refresh token."""
    token = get_refresh_token(email)
    return bool(token)
