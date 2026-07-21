"""
utils/visualizer.py
Visualization helpers — Attention Heatmap, Word Cloud, Charts
Twitter Hate Speech Detector
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud


# ── helper: fig → base64 PNG ──────────────────────────────────────────────────
def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


# ══════════════════════════════════════════════════════════════════
#  1. Attention Heatmap
# ══════════════════════════════════════════════════════════════════
def attention_heatmap(tokens: list, scores: list) -> str:
    """
    Generate a horizontal bar heatmap of attention scores per token.
    Returns base64-encoded PNG string.
    """
    if not tokens or not scores:
        return ""

    # Align lengths
    n = min(len(tokens), len(scores))
    tokens = tokens[:n]
    scores = np.array(scores[:n], dtype=float)

    # Normalise to [0,1]
    if scores.max() > 0:
        scores = scores / scores.max()

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 2.5),
                           facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = plt.cm.get_cmap("RdYlGn_r")
    colors = [cmap(s) for s in scores]

    bars = ax.barh(range(n), scores, color=colors, edgecolor="none", height=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(tokens, fontsize=10, color="white")
    ax.set_xlabel("Attention Weight", color="#aaaaaa", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.invert_yaxis()

    # value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", color="white", fontsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════
#  2. Token-level colour highlighting (HTML)
# ══════════════════════════════════════════════════════════════════
def highlight_tokens(tokens: list, scores: list) -> str:
    """
    Returns an HTML snippet where each token is wrapped in a <span>
    whose background colour reflects its attention score.
    High score → red; low score → transparent.
    """
    if not tokens or not scores:
        return ""

    n = min(len(tokens), len(scores))
    tokens = tokens[:n]
    scores = np.array(scores[:n], dtype=float)
    if scores.max() > 0:
        scores = scores / scores.max()

    cmap = plt.cm.get_cmap("YlOrRd")
    spans = []
    for token, score in zip(tokens, scores):
        r, g, b, _ = cmap(float(score))
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        alpha = max(0.15, float(score))
        style = (f"background:rgba({r},{g},{b},{alpha:.2f});"
                 f"color:#fff;padding:2px 5px;border-radius:4px;margin:2px;")
        spans.append(f'<span style="{style}" title="score: {score:.3f}">{token}</span>')

    return " ".join(spans)


# ══════════════════════════════════════════════════════════════════
#  3. Word Cloud
# ══════════════════════════════════════════════════════════════════
def generate_wordcloud(tweets: list) -> str:
    """
    Generate a word cloud from a list of tweet strings.
    Returns base64-encoded PNG string.
    """
    text = " ".join(tweets)
    if not text.strip():
        return ""

    wc = WordCloud(
        width=900, height=420,
        background_color="#0d1117",
        colormap="plasma",
        max_words=150,
        prefer_horizontal=0.85,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#0d1117")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════
#  4. Confusion Matrix
# ══════════════════════════════════════════════════════════════════
def confusion_matrix_chart(cm: np.ndarray, labels: list) -> str:
    """Render sklearn confusion matrix as heatmap → base64."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="magma",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#333",
        ax=ax,
        annot_kws={"size": 13, "color": "white"},
    )
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", color="#aaa")
    ax.set_ylabel("Actual",    color="#aaa")
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), rotation=20, color="white")
    plt.setp(ax.get_yticklabels(), rotation=0,  color="white")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════════
#  5. Training History Chart
# ══════════════════════════════════════════════════════════════════
def training_history_chart(history_dict: dict) -> str:
    """
    history_dict keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    Returns base64 PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines[["top","right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)

    # Accuracy
    axes[0].plot(epochs, history_dict.get("accuracy",     []), color="#00d2ff", lw=2, label="Train Accuracy")
    axes[0].plot(epochs, history_dict.get("val_accuracy", []), color="#ff6b6b", lw=2, linestyle="--", label="Val Accuracy")
    axes[0].set_title("Model Accuracy", color="white", fontsize=13)
    axes[0].set_xlabel("Epoch", color="#aaa")
    axes[0].set_ylabel("Accuracy", color="#aaa")
    axes[0].legend(framealpha=0.3, labelcolor="white")

    # Loss
    axes[1].plot(epochs, history_dict.get("loss",     []), color="#ffd700", lw=2, label="Train Loss")
    axes[1].plot(epochs, history_dict.get("val_loss", []), color="#ff4757", lw=2, linestyle="--", label="Val Loss")
    axes[1].set_title("Model Loss", color="white", fontsize=13)
    axes[1].set_xlabel("Epoch", color="#aaa")
    axes[1].set_ylabel("Loss", color="#aaa")
    axes[1].legend(framealpha=0.3, labelcolor="white")

    fig.tight_layout()
    return _fig_to_base64(fig)
