-- ============================================================
-- schema.sql
-- SQLite Database Schema
-- Twitter Hate Speech Detector
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- Users Table
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- Predictions Table
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    tweet TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_predictions_user
ON predictions(user_id);

CREATE INDEX IF NOT EXISTS idx_predictions_label
ON predictions(prediction);

CREATE INDEX IF NOT EXISTS idx_predictions_time
ON predictions(created_at);