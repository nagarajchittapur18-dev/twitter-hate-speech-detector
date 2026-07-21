"""
training/train_model.py
BiLSTM + Attention Model Training Script
Twitter Hate Speech Detector

Run: python training/train_model.py
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Gensim Word2Vec
from gensim.models import Word2Vec

# Project modules
from preprocessing.text_preprocessor import preprocess, get_tokens
from config import Config

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MAX_LEN       = Config.MAX_SEQUENCE_LEN   # 100
EMBED_DIM     = Config.EMBEDDING_DIM      # 100
VOCAB_SIZE    = Config.VOCAB_SIZE         # 20000
LSTM_UNITS    = 128
DENSE_UNITS   = 64
DROPOUT       = 0.4
BATCH_SIZE    = 64
EPOCHS        = 15
NUM_CLASSES   = 3

os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs("dataset",        exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  Custom Attention Layer
# ══════════════════════════════════════════════════════════════════
class AttentionLayer(layers.Layer):
    """Soft self-attention layer for sequence models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="att_context",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, features)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # (B, T, F)
        e = tf.tensordot(e, self.u, axes=1)                         # (B, T, 1)
        alpha = tf.nn.softmax(e, axis=1)                            # (B, T, 1)
        context = tf.reduce_sum(alpha * x, axis=1)                  # (B, F)
        return context, alpha

    def get_config(self):
        return super().get_config()


# ══════════════════════════════════════════════════════════════════
#  1. Load Dataset
# ══════════════════════════════════════════════════════════════════
def load_dataset():
    """
    Loads the hate-speech dataset. Downloads a built-in sample if the
    CSV is not present (for demo purposes).
    """
    if not os.path.exists(Config.DATASET_PATH):
        print("⚠️  Dataset not found — generating synthetic demo dataset …")
        _create_demo_dataset()

    df = pd.read_csv(Config.DATASET_PATH)
    print(f"✅ Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Normalise column names
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("tweet", "text", "comment", "content"):
            col_map[col] = "tweet"
        elif lc in ("class", "label", "category", "hate_speech"):
            col_map[col] = "label"
    df.rename(columns=col_map, inplace=True)

    if "tweet" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'tweet' and 'label' columns.")

    df.dropna(subset=["tweet", "label"], inplace=True)
    df["label"] = df["label"].astype(int)
    return df


def _create_demo_dataset():
    """Create a small synthetic dataset for demo / testing."""
    samples = [
        # Hate Speech (0)
        ("I hate those people they should not exist", 0),
        ("Kill all of them they are disgusting animals", 0),
        ("We need to exterminate this race", 0),
        ("They are subhuman and deserve nothing", 0),
        ("These immigrants are ruining everything", 0),
        ("Those people are vermin get rid of them", 0),
        ("All foreigners should be expelled now", 0),
        ("Disgusting filth they bring disease", 0),
        ("They are not human they are animals", 0),
        ("Burn their neighborhoods to the ground", 0),
        # Offensive Language (1)
        ("You are a complete idiot shut up", 1),
        ("What an absolute moron you are", 1),
        ("Stop being so damn stupid", 1),
        ("This is bullshit you are wrong", 1),
        ("You suck at everything loser", 1),
        ("Screw you and your opinion", 1),
        ("What a dumb piece of garbage", 1),
        ("You are freaking useless bro", 1),
        ("Total jerk move dude", 1),
        ("You are a clown stop talking", 1),
        # Neutral (2)
        ("I had a great day today", 2),
        ("The weather is nice this morning", 2),
        ("Just finished reading a good book", 2),
        ("Cooking dinner at home tonight", 2),
        ("Watching a movie with my family", 2),
        ("Excited for the weekend plans", 2),
        ("Good morning everyone have a nice day", 2),
        ("Just went for a walk in the park", 2),
        ("Learning Python is really fun", 2),
        ("Had coffee and feeling great", 2),
    ] * 100  # replicate to build ~3000 rows

    df = pd.DataFrame(samples, columns=["tweet", "label"])
    os.makedirs("dataset", exist_ok=True)
    df.to_csv(Config.DATASET_PATH, index=False)
    print(f"  Demo dataset saved → {Config.DATASET_PATH}")


# ══════════════════════════════════════════════════════════════════
#  2. Preprocessing
# ══════════════════════════════════════════════════════════════════
def prepare_data(df):
    print("🔄 Preprocessing tweets …")
    df["clean"] = df["tweet"].apply(preprocess)

    # Word2Vec tokenisation
    token_lists = df["tweet"].apply(get_tokens).tolist()

    print("🧠 Training Word2Vec embeddings …")
    w2v = Word2Vec(
        sentences=token_lists,
        vector_size=EMBED_DIM,
        window=5,
        min_count=1,
        workers=4,
        seed=SEED,
    )
    w2v.save(Config.W2V_PATH)
    print(f"  Word2Vec saved → {Config.W2V_PATH}")

    # Keras Tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean"].tolist())
    with open(Config.TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"  Tokenizer saved → {Config.TOKENIZER_PATH}")

    # Sequences
    sequences = tokenizer.texts_to_sequences(df["clean"].tolist())
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

    # Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(df["label"])
    y = keras.utils.to_categorical(y_enc, num_classes=NUM_CLASSES)
    with open(Config.LABEL_ENC_PATH, "wb") as f:
        pickle.dump(le, f)

    # Embedding matrix from Word2Vec
    word_index = tokenizer.word_index
    embed_matrix = np.zeros((min(VOCAB_SIZE, len(word_index) + 1), EMBED_DIM))
    for word, idx in word_index.items():
        if idx >= VOCAB_SIZE:
            continue
        if word in w2v.wv:
            embed_matrix[idx] = w2v.wv[word]

    return X, y, embed_matrix, tokenizer, le


# ══════════════════════════════════════════════════════════════════
#  3. Build Model
# ══════════════════════════════════════════════════════════════════
def build_model(embed_matrix):
    vocab_rows = embed_matrix.shape[0]

    inp = keras.Input(shape=(MAX_LEN,), name="input")

    # Embedding (pre-trained Word2Vec weights)
    x = layers.Embedding(
        input_dim=vocab_rows,
        output_dim=EMBED_DIM,
        weights=[embed_matrix],
        input_length=MAX_LEN,
        trainable=True,          # fine-tune
        name="embedding",
    )(inp)
    x = layers.SpatialDropout1D(0.2)(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name="bilstm",
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS // 2, return_sequences=True, dropout=0.3),
        name="bilstm2",
    )(x)

    # Attention
    context, alpha = AttentionLayer(name="attention")(x)

    # Dense head
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense1")(context)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=[out, alpha])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"output": "categorical_crossentropy", "attention": None},
        metrics={"output": ["accuracy"]},
    )
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════════
#  4. Train
# ══════════════════════════════════════════════════════════════════
def train():
    df = load_dataset()
    X, y, embed_matrix, tokenizer, le = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y.argmax(axis=1)
    )

    model = build_model(embed_matrix)

    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True, monitor="val_output_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=2, monitor="val_output_loss"),
        ModelCheckpoint(Config.MODEL_PATH, save_best_only=True, monitor="val_output_accuracy"),
    ]

    print("\n🚀 Training BiLSTM + Attention model …")
    history = model.fit(
        X_train, {"output": y_train, "attention": np.zeros((len(y_train), MAX_LEN, 1))},
        validation_data=(
            X_test,
            {"output": y_test, "attention": np.zeros((len(y_test), MAX_LEN, 1))},
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────────────────────────
    print("\n📊 Evaluating …")
    preds_out = model.predict(X_test, verbose=0)
    y_pred    = np.argmax(preds_out[0], axis=1)
    y_true    = np.argmax(y_test,       axis=1)

    label_names = ["Hate Speech", "Offensive Language", "Neutral"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    cm = confusion_matrix(y_true, y_pred)

    # ── Save charts ───────────────────────────────────────────────
    from utils.visualizer import confusion_matrix_chart, training_history_chart
    charts_dir = os.path.join(Config.STATIC_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    hist = history.history
    # map output_accuracy keys
    mapped = {
        "accuracy":     hist.get("output_accuracy",     hist.get("accuracy",     [])),
        "val_accuracy": hist.get("val_output_accuracy", hist.get("val_accuracy", [])),
        "loss":         hist.get("output_loss",         hist.get("loss",         [])),
        "val_loss":     hist.get("val_output_loss",     hist.get("val_loss",     [])),
    }

    import base64, io
    for name, b64 in [
        ("history.png", training_history_chart(mapped)),
        ("confusion_matrix.png", confusion_matrix_chart(cm, label_names)),
    ]:
        path = os.path.join(charts_dir, name)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  Chart saved → {path}")

    print(f"\n✅ Model saved → {Config.MODEL_PATH}")
    print("🎉 Training complete!")


if __name__ == "__main__":
    train()
