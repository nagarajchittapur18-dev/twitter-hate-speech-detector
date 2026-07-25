"""
app.py — Main Flask Application Entry Point
Twitter Hate Speech Detector
"""

import os
import sys
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from config import Config
from init_db import initialize_database
from database.db_manager import (
    create_user,
    get_user_by_username,
    verify_password,
    save_prediction,
    get_user_predictions,
    get_all_predictions,
    get_label_counts,
    get_total_count,
    get_recent_predictions,
    get_common_words,
)
from utils.helpers import predict_tweet, toxicity_level
from utils.visualizer import (
    attention_heatmap, highlight_tokens, generate_wordcloud
)
from api.predict_api import predict_bp

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
app.config["APP_CONFIG"] = Config  # expose to blueprints

app = Flask(__name__)
app.config.from_object(Config)
app.config["APP_CONFIG"] = Config

# Automatically create SQLite database if it doesn't exist
if not os.path.exists(Config.DATABASE_PATH):
    initialize_database()

# Blueprints
app.register_blueprint(predict_bp)


# Ensure static dirs exist
os.makedirs(os.path.join(Config.STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(Config.STATIC_DIR, "charts"),  exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  Auth helpers
# ══════════════════════════════════════════════════════════════════
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════
#  Landing Page
# ══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════
#  Auth Routes
# ══════════════════════════════════════════════════════════════════
@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm",  "").strip()

        if not all([username, email, password, confirm]):
            flash("All fields are required.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        else:
            success = create_user(username, email, password)
            if success:
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            else:
                flash("Username or email already exists.", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = get_user_by_username(username)
        if user and verify_password(user["password"], password):
            session["user_id"]   = user["id"]
            session["username"]  = user["username"]
            flash(f"Welcome back, {user['username']}! 🎉", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ══════════════════════════════════════════════════════════════════
#  Prediction Routes
# ══════════════════════════════════════════════════════════════════
@app.route("/home")
@login_required
def home():
    return render_template("home.html", username=session.get("username"))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    tweet = request.form.get("tweet", "").strip()

    if not tweet:
        flash("Please enter a tweet.", "warning")
        return redirect(url_for("home"))

    if len(tweet) > 500:
        flash("Tweet too long (max 500 chars).", "warning")
        return redirect(url_for("home"))

    result = predict_tweet(tweet, Config)

    # Persist prediction
    try:
        save_prediction(
            session["user_id"],
            tweet,
            result["label"],
            result["confidence"],
        )
    except Exception as e:
        app.logger.warning(f"DB save failed: {e}")

    # Visualizations
    heatmap_b64  = attention_heatmap(result["tokens"], result["attention"])
    highlights   = highlight_tokens(result["tokens"],  result["attention"])
    tox_level    = toxicity_level(result["label"], result["confidence"])

    return render_template(
        "result.html",
        tweet        = tweet,
        label        = result["label"],
        confidence   = result["confidence"],
        probabilities= result["probabilities"],
        tokens       = result["tokens"],
        attention    = result["attention"],
        heatmap      = heatmap_b64,
        highlights   = highlights,
        tox_level    = tox_level,
        username     = session.get("username"),
        label_color  = Config.LABEL_COLORS.get(result["label"], "#ffffff"),
    )


# ══════════════════════════════════════════════════════════════════
#  Dashboard
# ══════════════════════════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    label_counts  = get_label_counts()
    total         = get_total_count()
    recent_preds  = get_recent_predictions(10)
    user_history  = get_user_predictions(session["user_id"], 20)
    tweet_texts   = get_common_words(100)

    wordcloud_b64 = ""
    if tweet_texts:
        wordcloud_b64 = generate_wordcloud(tweet_texts)

    # Percentages
    def pct(n):
        return round((n / total * 100), 1) if total > 0 else 0

    stats = {
        "total":     total,
        "hate":      label_counts.get("Hate Speech",        0),
        "offensive": label_counts.get("Offensive Language", 0),
        "neutral":   label_counts.get("Neutral",            0),
        "hate_pct":      pct(label_counts.get("Hate Speech",        0)),
        "offensive_pct": pct(label_counts.get("Offensive Language", 0)),
        "neutral_pct":   pct(label_counts.get("Neutral",            0)),
    }

    # Static chart paths
    history_chart = url_for("static", filename="charts/history.png")  \
        if os.path.exists(os.path.join(Config.STATIC_DIR, "charts", "history.png")) else None
    cm_chart = url_for("static", filename="charts/confusion_matrix.png") \
        if os.path.exists(os.path.join(Config.STATIC_DIR, "charts", "confusion_matrix.png")) else None

    return render_template(
        "dashboard.html",
        stats        = stats,
        label_counts = label_counts,
        recent_preds = recent_preds,
        user_history = user_history,
        wordcloud    = wordcloud_b64,
        history_chart= history_chart,
        cm_chart     = cm_chart,
        username     = session.get("username"),
    )


# ══════════════════════════════════════════════════════════════════
#  Error Handlers
# ══════════════════════════════════════════════════════════════════
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html"), 500


# ══════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)