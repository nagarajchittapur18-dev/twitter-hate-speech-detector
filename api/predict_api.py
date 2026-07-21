"""
api/predict_api.py
REST API Blueprint — POST /api/predict
Twitter Hate Speech Detector
"""

from flask import Blueprint, request, jsonify, session
from utils.helpers import predict_tweet
from database.db_manager import save_prediction

predict_bp = Blueprint("predict_api", __name__, url_prefix="/api")


@predict_bp.route("/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body: {"tweet": "your tweet text"}

    Response:
    {
        "prediction":    "Hate Speech",
        "confidence":    95.2,
        "probabilities": {"Hate Speech": 95.2, "Offensive Language": 3.1, "Neutral": 1.7},
        "tokens":        ["word", ...],
        "attention":     [0.12, ...]
    }
    """
    from flask import current_app
    config = current_app.config["APP_CONFIG"]

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data  = request.get_json(silent=True) or {}
    tweet = data.get("tweet", "").strip()

    if not tweet:
        return jsonify({"error": "Field 'tweet' is required"}), 422

    if len(tweet) > 1000:
        return jsonify({"error": "Tweet too long (max 1000 chars)"}), 422

    result = predict_tweet(tweet, config)

    # Optionally persist if user is logged in
    if "user_id" in session:
        try:
            save_prediction(
                session["user_id"],
                tweet,
                result["label"],
                result["confidence"],
            )
        except Exception:
            pass   # non-blocking

    label_map = ["Hate Speech", "Offensive Language", "Neutral"]
    probs_dict = {
        label_map[i]: result["probabilities"][i]
        for i in range(len(label_map))
    }

    return jsonify({
        "prediction":    result["label"],
        "confidence":    result["confidence"],
        "probabilities": probs_dict,
        "tokens":        result["tokens"],
        "attention":     result["attention"],
    }), 200


@predict_bp.route("/status", methods=["GET"])
def api_status():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "service": "Twitter Hate Speech Detector API"}), 200
