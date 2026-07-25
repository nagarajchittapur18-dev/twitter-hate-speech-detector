"""
config.py — Application Configuration
Twitter Hate Speech Detector (SQLite Version)
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # ─────────────────────────────────────────────────────────────
    # Security
    # ─────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "hsd_super_secret_2024_!@#"
    )

    # ─────────────────────────────────────────────────────────────
    # SQLite Database
    # ─────────────────────────────────────────────────────────────
    DATABASE_PATH = os.path.join(BASE_DIR, "hate_speech.db")

    # ─────────────────────────────────────────────────────────────
    # Model Files
    # ─────────────────────────────────────────────────────────────
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    MODEL_PATH = os.path.join(
        MODEL_DIR,
        "bilstm_model.h5"
    )

    TOKENIZER_PATH = os.path.join(
        MODEL_DIR,
        "tokenizer.pkl"
    )

    W2V_PATH = os.path.join(
        MODEL_DIR,
        "word2vec.model"
    )

    LABEL_ENC_PATH = os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )

    # ─────────────────────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────────────────────
    DATASET_PATH = os.path.join(
        BASE_DIR,
        "dataset",
        "hate_speech.csv"
    )

    # ─────────────────────────────────────────────────────────────
    # NLP Parameters
    # ─────────────────────────────────────────────────────────────
    MAX_SEQUENCE_LEN = 100
    EMBEDDING_DIM = 100
    VOCAB_SIZE = 20000

    # ─────────────────────────────────────────────────────────────
    # Static / Uploads
    # ─────────────────────────────────────────────────────────────
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")

    # ─────────────────────────────────────────────────────────────
    # Labels
    # ─────────────────────────────────────────────────────────────
    LABELS = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neutral"
    }

    LABEL_COLORS = {
        "Hate Speech": "#ff4757",
        "Offensive Language": "#ffa502",
        "Neutral": "#2ed573",
    }