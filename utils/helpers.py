"""
utils/helpers.py
Common Utilities — model loading, prediction pipeline
Twitter Hate Speech Detector
"""

import os, sys, pickle, time
import numpy as np

# ── lazy-load TF to avoid slow import at module level ─────────────────────────
_model     = None
_tokenizer = None
_w2v       = None

# ── label map ─────────────────────────────────────────────────────────────────
LABELS = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}


def load_model_artifacts(config):
    """Load Keras model + tokenizer once and cache."""
    global _model, _tokenizer, _w2v

    if _model is not None:
        return _model, _tokenizer

    import tensorflow as tf
    from tensorflow import keras

    if not os.path.exists(config.MODEL_PATH):
        return None, None

    # Custom Attention layer must be registered before loading
    from training.train_model import AttentionLayer  # noqa: F401

    _model = keras.models.load_model(
        config.MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
    )

    with open(config.TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)

    return _model, _tokenizer


def predict_tweet(text: str, config) -> dict:
    """
    Full prediction pipeline.

    Returns
    -------
    dict with keys:
        label        : str
        confidence   : float (0-100)
        probabilities: list[float]
        tokens       : list[str]
        attention    : list[float]
    """
    from preprocessing.text_preprocessor import preprocess, get_tokens
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer = load_model_artifacts(config)

    if model is None or tokenizer is None:
        return {
            "label": "Model Not Ready",
            "confidence": 0.0,
            "probabilities": [0, 0, 0],
            "tokens": [],
            "attention": [],
        }

    clean_text = preprocess(text)
    tokens     = get_tokens(text)

    seq  = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=config.MAX_SEQUENCE_LEN, padding="post")

    # Model with attention returns [predictions, attention_weights]
    output = model.predict(padded, verbose=0)
    if isinstance(output, (list, tuple)) and len(output) == 2:
        probs, attention = output
        attention = attention[0].flatten().tolist()
    else:
        probs     = output
        attention = [1.0 / max(len(tokens), 1)] * len(tokens)

    probs      = probs[0].tolist()
    pred_idx   = int(np.argmax(probs))
    confidence = round(float(probs[pred_idx]) * 100, 2)
    label      = LABELS.get(pred_idx, "Unknown")

    # Align attention scores to tokens
    attn_scores = attention[: len(tokens)]
    if len(attn_scores) < len(tokens):
        attn_scores += [0.0] * (len(tokens) - len(attn_scores))

    return {
        "label":         label,
        "confidence":    confidence,
        "probabilities": [round(p * 100, 2) for p in probs],
        "tokens":        tokens,
        "attention":     attn_scores,
    }


def toxicity_level(label: str, confidence: float) -> str:
    """Map label + confidence → human-readable toxicity level."""
    if label == "Neutral":
        return "Safe"
    if label == "Hate Speech":
        return "Extreme" if confidence >= 80 else "High"
    # Offensive Language
    return "Moderate" if confidence >= 60 else "Low"


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if dt is None:
        return "N/A"
    return dt.strftime("%d %b %Y, %I:%M %p") if hasattr(dt, "strftime") else str(dt)
