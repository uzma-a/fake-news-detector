# TruthLens — AI Fake News Detector 🔍

> A full-stack AI-powered fake news detection system trained on Indian + International news sources.  
> Built with FastAPI, React.js, and a multi-model ML ensemble (LR + Naive Bayes + SVM).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![React](https://img.shields.io/badge/React-18-61DAFB)
![Accuracy](https://img.shields.io/badge/Accuracy-98.05%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-26%2C673%20articles-orange)

## 🎯 What It Does

TruthLens analyzes a news article and returns:
- ✅ **Verdict** — Fake or Real with confidence score
- 📊 **Model Comparison** — Logistic Regression, Naive Bayes, SVM side-by-side
- 🔤 **Word Influence** — Which words pushed the prediction and in which direction
- 🔴 **Highlighted Text** — Suspicious words highlighted inline in the article
- 🚨 **Pattern Detection** — 18 suspicious patterns (conspiracy language, urgency manipulation, Indian scams)

---

## 🏗️ Project Structure

```
fake-news-detector/
│
├── backend/                          # Python FastAPI backend
│   ├── model/                        # ML models and datasets
│   │   ├── all_models.joblib         # Trained LR + NB + SVM models
│   │   ├── tfidf_vectorizer.joblib   # TF-IDF vectorizer (50k features)
│   │   ├── feature_names.joblib      # Feature names for explainability
│   │   ├── model_results.joblib      # Accuracy metrics
│   │   ├── combined_dataset.csv      # 26,673 article training dataset
│   │   ├── indian_news.csv           # Indian RSS collected data
│   │   ├── Fake.csv                  # Kaggle fake news dataset
│   │   ├── True.csv                  # Kaggle real news dataset
│   │   ├── train_model.py            # Original single-model training
│   │   ├── multi_model.py            # Multi-model training script
│   │   ├── rss_collect.py            # Indian RSS data collector
│   │   ├── mix_datasets.py           # Dataset mixer script
│   │   └── try.py                    # HuggingFace dataset tester
│   │
│   ├── main.py                       # FastAPI app — main backend
│   ├── requirements.txt              # Python dependencies
│   └── Dockerfile                    # Backend Docker config
│
├── frontend/                         # React.js frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Detector.jsx          # Main UI component
│   │   │   └── Detector.css          # Dark editorial styles
│   │   ├── App.jsx                   # Root component
│   │   └── main.jsx                  # React entry point
│   ├── public/
│   ├── index.html                    # HTML entry point
│   ├── package.json                  # Node dependencies
│   ├── vite.config.js                # Vite configuration
│   └── Dockerfile                    # Frontend Docker config
│
├── docker-compose.yml                # Full stack Docker setup
└── README.md                         # This file
```

---

## 🤖 ML Architecture

### Preprocessing Pipeline
```
Raw Text
   ↓
Lowercase → Remove URLs/HTML → Remove special chars
   ↓
Tokenize (NLTK word_tokenize)
   ↓
Remove Stopwords + Porter Stemming
   ↓
TF-IDF Vectorization (50,000 features, bigrams)
   ↓
Multi-Model Prediction
```

### Model Performance

| Model | Accuracy | Role |
|-------|----------|------|
| Logistic Regression | 97.39% | Explainability (word influence) |
| Naive Bayes | 95.18% | Fast probabilistic baseline |
| **SVM (LinearSVC)** | **98.05%** | **Primary verdict model** |

### Dataset
| Source | Type | Count |
|--------|------|-------|
| GonzaloA/fake_news (HuggingFace) | International | 24,353 |
| BoomLive RSS | Indian Fake (fact-checked) | ~59 |
| AltNews RSS | Indian Fake (fact-checked) | ~10 |
| The Hindu RSS | Indian Real | ~60 |
| NDTV / Indian Express RSS | Indian Real | ~93 |
| **Total** | **Combined** | **26,673** |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend Setup
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train models (first time only)
cd model
python multi_model.py
cd ..

# Start backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup
```bash
# New terminal
cd frontend
npm install
npm run dev
```

### Open in Browser
```
Frontend → http://localhost:5173
API Docs → http://localhost:8000/docs
Health   → http://localhost:8000/health
```

---

## 📡 API Reference

### `POST /predict`

**Request:**
```json
{
  "text": "BREAKING: Secret government conspiracy exposed..."
}
```

**Response:**
```json
{
  "prediction": "Fake",
  "confidence": 0.9609,
  "fake_probability": 0.9609,
  "real_probability": 0.0391,
  "word_count": 66,
  "processing_time_ms": 6.76,
  "influential_words": [
    { "word": "break", "score": 0.312, "direction": "fake" },
    { "word": "govern", "score": 0.189, "direction": "real" }
  ],
  "suspicious_patterns": [
    "ALL-CAPS sensational headline",
    "Conspiracy terminology",
    "Viral urgency manipulation"
  ],
  "explanation": "This article shows strong indicators of misinformation...",
  "model_predictions": {
    "Logistic Regression": { "prediction": "Fake", "fake_probability": 0.84, "real_probability": 0.16 },
    "Naive Bayes":         { "prediction": "Fake", "fake_probability": 0.83, "real_probability": 0.17 },
    "SVM":                 { "prediction": "Fake", "fake_probability": 1.00, "real_probability": 0.00 }
  }
}
```

### `GET /health`
Returns model status, loaded model names, and accuracy metrics.

---

## 🇮🇳 India-Specific Features

- Trained on **BoomLive** and **AltNews** fact-checked Indian articles
- Detects **WhatsApp-style** fake news patterns
- Indian political terms (BJP, Modi, Lok Sabha, RBI) correctly classified
- Pattern detection covers Indian-specific scams:
  - Free government scheme hoaxes
  - Anonymous IAS/officer source claims
  - "Forward karo" urgency manipulation
  - False deadline scarcity claims

---

## 🧠 Why Not Just Use ChatGPT?

| Factor | LLM (ChatGPT) | TruthLens |
|--------|--------------|-----------|
| Cost | $300+/month at scale | **Free** |
| Speed | 3–10 seconds | **6–150ms** |
| Explainability | Black box | **Word-level scores** |
| Privacy | Data leaves device | **Fully local** |
| India Focus | Generic | **Indian data trained** |
| Offline | ❌ | **✅** |

---

## 🛠️ Tech Stack

**Backend:** Python 3.11, FastAPI, scikit-learn, NLTK, joblib, numpy, pandas  
**Frontend:** React.js 18, Vite, CSS3  
**ML:** TF-IDF + Logistic Regression + Naive Bayes + SVM  
**Data:** HuggingFace Datasets, BoomLive RSS, AltNews RSS, The Hindu RSS  
**DevOps:** Docker, Docker Compose  

---

## 📈 Results

```
✅ Accuracy:              98.05% (SVM)
✅ Processing Time:       6–150ms
✅ Manual Test Accuracy:  7/7 correct
✅ Indian News:           54% → 91% after Indian dataset
✅ Dataset Size:          26,673 articles
✅ Suspicious Patterns:   18 (global + India-specific)
```

---

## 🔮 Future Plans

- [ ] Deploy on Vercel + Render
- [ ] Add BERT transformer model
- [ ] Hindi / Hinglish language support
- [ ] WhatsApp bot integration
- [ ] Chrome browser extension
- [ ] Daily automated RSS retraining
- [ ] LIME explainability integration

---

## 👩‍💻 Author

Built by **Uzma** — for educational use and job portfolio  
For educational purposes only. Always verify news with multiple credible sources.

---

## ⭐ If you found this useful, please star the repo!
