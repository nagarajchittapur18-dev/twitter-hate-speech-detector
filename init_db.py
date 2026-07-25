"""
init_db.py
Initialize SQLite Database
Twitter Hate Speech Detector
"""

import sqlite3
from config import Config


def initialize_database():
    """Create SQLite database and tables."""

    conn = sqlite3.connect(Config.DATABASE_PATH)

    try:
        with open("schema.sql", "r", encoding="utf-8") as f:
            schema = f.read()

        conn.executescript(schema)
        conn.commit()

        print("=" * 60)
        print(" SQLite database created successfully!")
        print(f"Database Location: {Config.DATABASE_PATH}")
        print("Tables created successfully.")
        print("=" * 60)

    except Exception as e:
        print("Error creating database:")
        print(e)

    finally:
        conn.close()


if __name__ == "__main__":
    initialize_database()