"""SQLite helpers for storing users and study sessions."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime, timezone
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "data" / "pbl.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            case_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            pre_score REAL,
            post_score REAL,
            evaluation_json TEXT,
            advice_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()


def seed_users(users: Iterable[tuple[str, str]] | None = None) -> None:
    defaults = users or (
        ("student_lee", "pbl123"),
        ("student_wang", "med2024"),
    )
    conn = get_connection()
    cur = conn.cursor()
    for username, password in defaults:
        cur.execute(
            "INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)",
            (username, password),
        )
    conn.commit()
    conn.close()


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username FROM users WHERE username = ? AND password = ?",
        (username, password),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row["id"], "username": row["username"]}


def fetch_user_history(user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT case_id, pre_score, post_score, evaluation_json, advice_json, started_at, ended_at
        FROM study_sessions
        WHERE user_id = ?
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    history: List[Dict[str, Any]] = []
    for row in rows:
        history.append({
            "case_id": row["case_id"],
            "pre_score": row["pre_score"],
            "post_score": row["post_score"],
            "evaluation": json.loads(row["evaluation_json"]) if row["evaluation_json"] else None,
            "advice": json.loads(row["advice_json"]) if row["advice_json"] else None,
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
        })
    return history


def record_study_session(
    user_id: int,
    case_id: str,
    started_at: str,
    ended_at: str,
    pre_score: float,
    post_score: float,
    evaluation: Dict[str, Any],
    advice: Dict[str, Any],
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO study_sessions (
            user_id, case_id, started_at, ended_at, pre_score, post_score, evaluation_json, advice_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            case_id,
            started_at,
            ended_at,
            float(pre_score),
            float(post_score),
            json.dumps(evaluation, ensure_ascii=False),
            json.dumps(advice, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def record_feedback(user_id: int, content: str, created_at: Optional[str] = None) -> None:
    conn = get_connection()
    cur = conn.cursor()
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        INSERT INTO feedbacks (user_id, content, created_at)
        VALUES (?, ?, ?)
        """,
        (user_id, content.strip(), timestamp),
    )
    conn.commit()
    conn.close()


def create_session_token(user_id: int, token: str, expires_at: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    cur.execute(
        """
        INSERT INTO sessions (user_id, token, expires_at)
        VALUES (?, ?, ?)
        """,
        (user_id, token, expires_at),
    )
    conn.commit()
    conn.close()


def fetch_user_by_session_token(token: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sessions.user_id, sessions.expires_at, users.username
        FROM sessions
        JOIN users ON users.id = sessions.user_id
        WHERE sessions.token = ?
        """,
        (token,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    expires_raw = row["expires_at"]
    try:
        expires_at = datetime.fromisoformat(expires_raw)
    except Exception:
        expires_at = None
    now = datetime.now(timezone.utc)
    if expires_at is None:
        return None
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < now:
        delete_session_token(token)
        return None
    return {"id": row["user_id"], "username": row["username"]}


def delete_session_token(token: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()
