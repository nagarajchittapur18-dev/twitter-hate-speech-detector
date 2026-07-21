"""
preprocessing/text_preprocessor.py
Full Social-Media NLP Preprocessing Pipeline
Twitter Hate Speech Detector
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Download required NLTK data (first-run only) ──────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

# ── Singleton resources ───────────────────────────────────────────────────────
_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS  = set(stopwords.words("english"))

# Keep negations — they flip sentiment
_KEEP = {"no", "not", "nor", "never", "neither", "nobody", "nothing", "nowhere"}
_STOP_WORDS -= _KEEP

# ── Regex patterns ────────────────────────────────────────────────────────────
_URL_PATTERN       = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN   = re.compile(r"@\w+")
_HASHTAG_PATTERN   = re.compile(r"#(\w+)")          # keep the word, drop the #
_EMOJI_PATTERN     = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u200d"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\ufe0f"
    u"\u3030"
    "]+",
    flags=re.UNICODE,
)
_PUNCT_PATTERN     = re.compile(r"[%s]" % re.escape(string.punctuation))
_EXTRA_SPACE_PAT   = re.compile(r"\s+")
_REPEAT_CHAR_PAT   = re.compile(r"(.)\1{2,}")       # "loooool" → "lol"
_HTML_TAG_PATTERN  = re.compile(r"<.*?>")
_NUMBER_PATTERN    = re.compile(r"\b\d+\b")


def preprocess(text: str, return_tokens: bool = False):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    text         : raw tweet string
    return_tokens: if True returns list[str], else joined string

    Returns
    -------
    str | list[str]
    """
    if not isinstance(text, str) or not text.strip():
        return [] if return_tokens else ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = _HTML_TAG_PATTERN.sub(" ", text)

    # 3. Remove URLs
    text = _URL_PATTERN.sub(" ", text)

    # 4. Remove @mentions
    text = _MENTION_PATTERN.sub(" ", text)

    # 5. Clean hashtags (keep the word)
    text = _HASHTAG_PATTERN.sub(r"\1", text)

    # 6. Remove emojis
    text = _EMOJI_PATTERN.sub(" ", text)

    # 7. Reduce repeated characters
    text = _REPEAT_CHAR_PAT.sub(r"\1\1", text)

    # 8. Remove punctuation
    text = _PUNCT_PATTERN.sub(" ", text)

    # 9. Remove standalone numbers
    text = _NUMBER_PATTERN.sub(" ", text)

    # 10. Normalize whitespace
    text = _EXTRA_SPACE_PAT.sub(" ", text).strip()

    # 11. Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 12. Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    # 13. Lemmatize
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens if return_tokens else " ".join(tokens)


def get_tokens(text: str):
    """Return token list for a raw tweet."""
    return preprocess(text, return_tokens=True)


def batch_preprocess(texts):
    """Preprocess a list/Series of tweets."""
    return [preprocess(t) for t in texts]
