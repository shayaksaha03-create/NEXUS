"""
NEXUS AI - User Account Manager
Handles user registration, authentication, and chat history persistence.
Uses SQLite for storage and hashlib+secrets for password hashing.
"""

import sqlite3
import hashlib
import secrets
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR
from utils.logger import get_logger
logger = get_logger("user_manager")

DB_PATH = Path(DATA_DIR) / "users.db"


class UserManager:
    """Manages user accounts and chat history with SQLite storage."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._db_lock = threading.Lock()
        self._init_db()
        logger.info(f"UserManager initialized — DB at {DB_PATH}")

    # ── Database Setup ──

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        display_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        emotion TEXT DEFAULT 'neutral',
                        intensity REAL DEFAULT 0.5,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_chat_user_time
                        ON chat_history(user_id, timestamp DESC);
                """)
                conn.commit()

                # ── Schema migrations ──
                for col, col_type in [("profile_picture", "TEXT"), ("bio", "TEXT")]:
                    try:
                        conn.execute(f"ALTER TABLE users ADD COLUMN {col} {col_type}")
                        conn.commit()
                        logger.info(f"Added column users.{col}")
                    except sqlite3.OperationalError:
                        pass  # Column already exists

                logger.info("User database tables ready")
            finally:
                conn.close()

    # ── Password Hashing ──

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100_000
        ).hex()

    # ── User CRUD ──

    def create_user(self, username: str, password: str,
                    display_name: str = "") -> Dict[str, Any]:
        """
        Create a new user account.
        Returns user dict on success, raises ValueError on duplicate.
        """
        username = username.strip()
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if not password or len(password) < 4:
            raise ValueError("Password must be at least 4 characters")

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        display = display_name.strip() or username

        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, display_name) "
                    "VALUES (?, ?, ?, ?)",
                    (username, password_hash, salt, display)
                )
                conn.commit()
                user_id = conn.execute(
                    "SELECT id FROM users WHERE username = ?", (username,)
                ).fetchone()["id"]

                logger.info(f"User created: {username} (id={user_id})")
                return {
                    "id": user_id,
                    "username": username,
                    "display_name": display,
                }
            except sqlite3.IntegrityError:
                raise ValueError(f"Username '{username}' already exists")
            finally:
                conn.close()

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user. Returns user dict or None."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username.strip(),)
                ).fetchone()
                if not row:
                    return None

                expected = self._hash_password(password, row["salt"])
                if expected != row["password_hash"]:
                    return None

                # Update last login
                conn.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (row["id"],)
                )
                conn.commit()

                logger.info(f"User authenticated: {username}")
                return {
                    "id": row["id"],
                    "username": row["username"],
                    "display_name": row["display_name"] or row["username"],
                }
            finally:
                conn.close()

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT id, username, display_name, created_at FROM users WHERE id = ?",
                    (user_id,)
                ).fetchone()
                if not row:
                    return None
                return dict(row)
            finally:
                conn.close()

    def get_full_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get full user profile including bio and avatar."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT id, username, display_name, bio, profile_picture, "
                    "created_at, last_login FROM users WHERE id = ?",
                    (user_id,)
                ).fetchone()
                if not row:
                    return None
                return dict(row)
            finally:
                conn.close()

    def update_profile(self, user_id: int, display_name: str = "",
                       bio: str = "") -> Dict[str, Any]:
        """Update user display_name and bio."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE users SET display_name = ?, bio = ? WHERE id = ?",
                    (display_name.strip(), bio.strip(), user_id)
                )
                conn.commit()
                logger.info(f"Profile updated for user {user_id}")
                return self.get_full_profile(user_id) or {}
            finally:
                conn.close()

    def change_password(self, user_id: int, old_password: str,
                        new_password: str) -> bool:
        """Change password after verifying old password. Returns True on success."""
        if not new_password or len(new_password) < 4:
            raise ValueError("New password must be at least 4 characters")

        with self._db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT password_hash, salt FROM users WHERE id = ?",
                    (user_id,)
                ).fetchone()
                if not row:
                    raise ValueError("User not found")

                expected = self._hash_password(old_password, row["salt"])
                if expected != row["password_hash"]:
                    raise ValueError("Current password is incorrect")

                new_salt = secrets.token_hex(16)
                new_hash = self._hash_password(new_password, new_salt)
                conn.execute(
                    "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                    (new_hash, new_salt, user_id)
                )
                conn.commit()
                logger.info(f"Password changed for user {user_id}")
                return True
            finally:
                conn.close()

    def update_avatar(self, user_id: int, base64_data: str) -> bool:
        """Store a base64-encoded profile picture."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE users SET profile_picture = ? WHERE id = ?",
                    (base64_data, user_id)
                )
                conn.commit()
                logger.info(f"Avatar updated for user {user_id}")
                return True
            finally:
                conn.close()

    # ── Chat History ──

    def save_message(self, user_id: int, role: str, content: str,
                     emotion: str = "neutral", intensity: float = 0.5):
        """Save a chat message to the database."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO chat_history (user_id, role, content, emotion, intensity) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, role, content, emotion, intensity)
                )
                conn.commit()
            finally:
                conn.close()

    def get_chat_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent chat history for a user, oldest first."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT role, content, emotion, intensity, timestamp "
                    "FROM chat_history WHERE user_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
                # Reverse so oldest is first
                return [dict(r) for r in reversed(rows)]
            finally:
                conn.close()

    def clear_chat_history(self, user_id: int):
        """Clear all chat history for a user."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM chat_history WHERE user_id = ?", (user_id,)
                )
                conn.commit()
                logger.info(f"Chat history cleared for user {user_id}")
            finally:
                conn.close()


# Global instance
user_manager = UserManager()
