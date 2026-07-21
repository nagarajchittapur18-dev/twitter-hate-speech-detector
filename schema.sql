-- ============================================================
-- schema.sql — MySQL Database Schema
-- Twitter Hate Speech Detector
-- ============================================================

CREATE DATABASE IF NOT EXISTS hate_speech_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE hate_speech_db;

-- ── Users Table ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(80)  NOT NULL UNIQUE,
    email      VARCHAR(120) NOT NULL UNIQUE,
    password   VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ── Predictions Table ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT NOT NULL,
    tweet       TEXT        NOT NULL,
    prediction  VARCHAR(50) NOT NULL,
    confidence  FLOAT       NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ── Indexes ───────────────────────────────────────────────────
CREATE INDEX idx_predictions_user   ON predictions(user_id);
CREATE INDEX idx_predictions_label  ON predictions(prediction);
CREATE INDEX idx_predictions_time   ON predictions(created_at);
