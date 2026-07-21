"""
database/db_manager.py
MySQL Database Manager — CRUD helpers
Twitter Hate Speech Detector
"""

from flask_mysqldb import MySQL
import MySQLdb
from werkzeug.security import generate_password_hash, check_password_hash

mysql = MySQL()  # shared instance — init_app() called in app.py


# ══════════════════════════════════════════════════════════════════
#  USER helpers
# ══════════════════════════════════════════════════════════════════

def create_user(username: str, email: str, password: str) -> bool:
    """
    Insert a new user. Password is bcrypt-hashed.
    Returns True on success, False if username/email already exists.
    """
    hashed = generate_password_hash(password)
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed),
        )
        mysql.connection.commit()
        cur.close()
        return True
    except MySQLdb.IntegrityError:
        return False


def get_user_by_username(username: str):
    """Return user row dict or None."""
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    return user


def get_user_by_id(user_id: int):
    """Return user row dict or None."""
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    return user


def verify_password(stored_hash: str, password: str) -> bool:
    return check_password_hash(stored_hash, password)


# ══════════════════════════════════════════════════════════════════
#  PREDICTION helpers
# ══════════════════════════════════════════════════════════════════

def save_prediction(user_id: int, tweet: str, prediction: str, confidence: float):
    """Persist a prediction row."""
    cur = mysql.connection.cursor()
    cur.execute(
        """INSERT INTO predictions (user_id, tweet, prediction, confidence)
           VALUES (%s, %s, %s, %s)""",
        (user_id, tweet, prediction, round(confidence, 2)),
    )
    mysql.connection.commit()
    cur.close()


def get_user_predictions(user_id: int, limit: int = 50):
    """Return recent predictions for a user."""
    cur = mysql.connection.cursor()
    cur.execute(
        """SELECT * FROM predictions WHERE user_id = %s
           ORDER BY created_at DESC LIMIT %s""",
        (user_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def get_all_predictions(limit: int = 500):
    """Return all predictions for dashboard analytics."""
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s", (limit,)
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def get_label_counts():
    """
    Returns dict: {'Hate Speech': N, 'Offensive Language': M, 'Neutral': K}
    """
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT prediction, COUNT(*) AS cnt FROM predictions GROUP BY prediction"
    )
    rows = cur.fetchall()
    cur.close()
    result = {"Hate Speech": 0, "Offensive Language": 0, "Neutral": 0}
    for row in rows:
        result[row["prediction"]] = row["cnt"]
    return result


def get_total_count() -> int:
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM predictions")
    row = cur.fetchone()
    cur.close()
    return row["cnt"] if row else 0


def get_recent_predictions(limit: int = 10):
    cur = mysql.connection.cursor()
    cur.execute(
        """SELECT p.tweet, p.prediction, p.confidence, p.created_at, u.username
           FROM predictions p
           JOIN users u ON p.user_id = u.id
           ORDER BY p.created_at DESC LIMIT %s""",
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def get_common_words(limit: int = 50):
    """Return all tweet texts for word-cloud generation."""
    cur = mysql.connection.cursor()
    cur.execute("SELECT tweet FROM predictions ORDER BY created_at DESC LIMIT 200")
    rows = cur.fetchall()
    cur.close()
    return [r["tweet"] for r in rows]
