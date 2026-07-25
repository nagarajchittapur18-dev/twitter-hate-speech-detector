"""
database/db_manager.py
SQLite Database Manager — CRUD helpers
Twitter Hate Speech Detector
"""

import sqlite3
from config import Config
from werkzeug.security import generate_password_hash, check_password_hash


# ------------------------------------------------------------------
# Database Connection
# ------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Dummy object so old imports don't immediately fail.
# (We'll remove it from app.py in the next step.)
mysql = None


# ================================================================
# USER HELPERS
# ================================================================

def create_user(username: str, email: str, password: str) -> bool:
    """
    Create a new user.
    Returns True if successful, False if username/email already exists.
    """
    hashed = generate_password_hash(password)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?)
            """,
            (username, email, hashed),
        )

        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()


def get_user_by_username(username: str):
    """
    Return user as dictionary or None.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username=?",
        (username,),
    )

    row = cur.fetchone()
    conn.close()

    return dict(row) if row else None


def get_user_by_id(user_id: int):
    """
    Return user as dictionary or None.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE id=?",
        (user_id,),
    )

    row = cur.fetchone()
    conn.close()

    return dict(row) if row else None


def verify_password(stored_hash: str, password: str) -> bool:
    return check_password_hash(stored_hash, password)


# ================================================================
# PREDICTION HELPERS
# ================================================================

def save_prediction(
    user_id: int,
    tweet: str,
    prediction: str,
    confidence: float,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO predictions
        (user_id, tweet, prediction, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            tweet,
            prediction,
            round(confidence, 2),
        ),
    )

    conn.commit()
    conn.close()


def get_user_predictions(user_id: int, limit: int = 50):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id=?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_all_predictions(limit: int = 500):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_label_counts():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT prediction,
               COUNT(*) AS cnt
        FROM predictions
        GROUP BY prediction
        """
    )

    rows = cur.fetchall()
    conn.close()

    result = {
        "Hate Speech": 0,
        "Offensive Language": 0,
        "Neutral": 0,
    }

    for row in rows:
        result[row["prediction"]] = row["cnt"]

    return result


def get_total_count() -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM predictions
        """
    )

    row = cur.fetchone()
    conn.close()

    return row["cnt"] if row else 0


def get_recent_predictions(limit: int = 10):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            p.tweet,
            p.prediction,
            p.confidence,
            p.created_at,
            u.username
        FROM predictions p
        JOIN users u
            ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_common_words(limit: int = 50):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT tweet
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()

    return [row["tweet"] for row in rows]