# Twitter Hate Speech Detector 🛡️

> **AI-Powered Twitter Hate Speech Detection using NLP and Deep Learning**
> BiLSTM + Attention Mechanism | Word2Vec | Flask | MySQL | Dark UI

---

## 📋 Project Overview

A full-stack web application that classifies tweets into:
| Class | Label | Description |
|-------|-------|-------------|
| 0 | 🔥 Hate Speech | Targets individuals/groups based on protected characteristics |
| 1 | ⚠️ Offensive Language | Profane or derogatory content |
| 2 | ✅ Neutral | Safe, clean content |

---

## 🚀 Features

- ✅ Full Social-Media NLP Preprocessing (13 steps)
- ✅ Word2Vec Embeddings (Gensim)
- ✅ BiLSTM + Custom Attention Layer (TensorFlow/Keras)
- ✅ Attention Heatmap & Toxic Word Highlighting
- ✅ Analytics Dashboard (Chart.js charts + Word Cloud)
- ✅ User Authentication (Register/Login/Logout)
- ✅ Prediction History per user
- ✅ REST API (`POST /api/predict`)
- ✅ Dark Glassmorphism UI (Bootstrap 5)
- ✅ Mobile Responsive Design

---

## 🗂️ Folder Structure

```
Twitter Hate Speech Detector/
├── app.py                    # Flask entry point
├── config.py                 # Configuration (DB, model paths, secrets)
├── requirements.txt          # Python dependencies
├── schema.sql                # MySQL DDL
├── README.md
│
├── model/                    # Saved model artifacts (auto-generated)
│   ├── bilstm_model.h5
│   ├── tokenizer.pkl
│   ├── word2vec.model
│   └── label_encoder.pkl
│
├── dataset/
│   └── hate_speech.csv       # Training dataset
│
├── preprocessing/
│   └── text_preprocessor.py  # NLP pipeline
│
├── training/
│   └── train_model.py        # Model training script
│
├── utils/
│   ├── visualizer.py         # Heatmaps, word cloud, charts
│   └── helpers.py            # Model loading, prediction pipeline
│
├── database/
│   └── db_manager.py         # MySQL CRUD operations
│
├── api/
│   └── predict_api.py        # REST API blueprint
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── charts/               # Training charts (auto-generated)
│
└── templates/
    ├── base.html
    ├── index.html             # Landing page
    ├── home.html              # Prediction input
    ├── result.html            # Results + heatmap
    ├── dashboard.html         # Analytics
    ├── login.html
    ├── register.html
    ├── 404.html
    └── 500.html
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask 2.3 |
| Database | MySQL 8.0 |
| Deep Learning | TensorFlow 2.13, Keras |
| NLP | NLTK, Gensim Word2Vec |
| Visualization | Matplotlib, WordCloud, Chart.js |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Authentication | Werkzeug bcrypt |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.10+
- MySQL 8.0+
- pip

### 2. Clone / Open Project
```bash
cd "Twitter Hate Speech Detector"
```

### 3. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Database

Open `config.py` and update:
```python
MYSQL_USER     = "root"
MYSQL_PASSWORD = "your_password"   # ← set your MySQL password
MYSQL_DB       = "hate_speech_db"
```

Then create the database:
```bash
mysql -u root -p < schema.sql
```

### 6. Download NLTK Data
```bash
python -c "import nltk; nltk.download('all')"
```

### 7. Add Dataset

**Option A — Use your own dataset:**
Place `hate_speech.csv` in the `dataset/` folder.

Required columns:
- `tweet` — raw tweet text
- `label` — integer class (0=Hate Speech, 1=Offensive, 2=Neutral)

**Recommended Dataset:**
[Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)
Download `labeled_data.csv`, rename columns to `tweet` and `label`.

**Option B — Auto-generated demo dataset:**
If no CSV is found, `train_model.py` automatically creates a synthetic demo dataset.

### 8. Train the Model
```bash
python training/train_model.py
```

This will:
- Preprocess all tweets
- Train Word2Vec embeddings
- Train BiLSTM + Attention model
- Save model artifacts to `model/`
- Save training charts to `static/charts/`

Training time: ~5–15 minutes (GPU recommended)

### 9. Run the Application
```bash
python app.py
```

Open browser → `http://localhost:5000`

---

## 🔌 REST API

### `POST /api/predict`

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "I love this community!"}'
```

**Response:**
```json
{
  "prediction":    "Neutral",
  "confidence":    92.4,
  "probabilities": {
    "Hate Speech":        1.2,
    "Offensive Language": 6.4,
    "Neutral":            92.4
  },
  "tokens":    ["love", "community"],
  "attention": [0.32, 0.68]
}
```

### `GET /api/status`
```json
{"status": "ok", "service": "Twitter Hate Speech Detector API"}
```

---

## 📊 Model Architecture

```
Input Tweet
    ↓
NLP Preprocessing (13 steps)
    ↓
Word2Vec Embedding Layer (100-dim)
    ↓
SpatialDropout1D
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Custom Attention Layer → [context_vector, attention_weights]
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.4) + BatchNormalization
    ↓
Dense (3, Softmax)
    ↓
Prediction: Hate Speech | Offensive Language | Neutral
```

---

## 🧹 NLP Preprocessing Pipeline

| Step | Operation |
|------|-----------|
| 1 | Lowercase conversion |
| 2 | HTML tag removal |
| 3 | URL removal |
| 4 | @mention removal |
| 5 | Hashtag cleaning (keep word) |
| 6 | Emoji removal |
| 7 | Repeated character reduction |
| 8 | Punctuation removal |
| 9 | Standalone number removal |
| 10 | Whitespace normalization |
| 11 | NLTK tokenization |
| 12 | Stopword removal (keeping negations) |
| 13 | WordNet lemmatization |

---

## 🌐 Application Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Landing | Hero, features, API preview |
| `/home` | Detect | Tweet input form |
| `/predict` | POST → Result | Prediction + heatmap |
| `/dashboard` | Dashboard | Analytics + charts |
| `/login` | Login | Authentication |
| `/register` | Register | New account |
| `/logout` | — | Clear session |
| `/api/predict` | API | REST endpoint |

---

## 🚢 Deployment

### Railway / Render

1. Set environment variables:
```
SECRET_KEY=your_secret_key
MYSQL_HOST=your_db_host
MYSQL_USER=your_db_user
MYSQL_PASSWORD=your_db_password
MYSQL_DB=hate_speech_db
```

2. Add `Procfile`:
```
web: python app.py
```

3. Push to GitHub and connect to Railway/Render.

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## 👨‍💻 Author

**Twitter Hate Speech Detector**
Built for MCA Final Year Project | NLP + Deep Learning Portfolio

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.
