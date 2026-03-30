"""
Fake News Detector - FastAPI Backend
=====================================
Serves the ML model and provides prediction, confidence, and explainability.
"""

import os
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ── Model globals ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
tfidf_vectorizer = None
all_models       = None
model_results    = None
feature_names    = None


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── Lifespan: load models on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tfidf_vectorizer, all_models, model_results, feature_names
    logger.info("Loading ML models...")

    tfidf_path   = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    models_path  = os.path.join(MODEL_DIR, "all_models.joblib")
    results_path = os.path.join(MODEL_DIR, "model_results.joblib")
    fn_path      = os.path.join(MODEL_DIR, "feature_names.joblib")

    if not os.path.exists(models_path):
        logger.warning("Multi-models not found. Training now...")
        import subprocess, sys
        train_script = os.path.join(MODEL_DIR, "multi_model.py")
        subprocess.run([sys.executable, train_script], cwd=MODEL_DIR, check=True)

    tfidf_vectorizer = joblib.load(tfidf_path)
    all_models       = joblib.load(models_path)
    model_results    = joblib.load(results_path) if os.path.exists(results_path) else {}
    feature_names    = joblib.load(fn_path) if os.path.exists(fn_path) else None

    logger.info(f"✅ {len(all_models)} models loaded!")
    yield
    logger.info("Shutting down...")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detector API",
    description="Detects fake news using NLP + Multi-Model ML",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=20, description="News article text to analyze")

    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v


class WordInfluence(BaseModel):
    word: str
    score: float
    direction: str  # "fake" or "real"


class ModelResult(BaseModel):
    prediction: str
    fake_probability: float
    real_probability: float


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    influential_words: list[WordInfluence]
    explanation: str
    processing_time_ms: float
    word_count: int
    suspicious_patterns: list[str]
    model_predictions: dict[str, ModelResult] = {}


# ── Suspicious patterns (NO duplicates) ───────────────────────────────────────
SUSPICIOUS_PATTERNS = [
    # General fake news patterns
    (r"\b(BREAKING|URGENT|SHOCKING|EXCLUSIVE)\b",       "ALL-CAPS sensational headline"),
    (r"they don.t want you to know",                     "Conspiracy phrasing"),
    (r"mainstream media (refuses|won.t|ignores)",        "Media distrust framing"),
    (r"\b(elites?|globalists?|deep state)\b",            "Conspiracy terminology"),
    (r"share (this|before it.s deleted|before they delete)", "Viral urgency manipulation"),
    (r"doctors? (hate|don.t want)",                      "Pseudoscience framing"),
    (r"(secret|hidden) (cure|truth|agenda|plot)",        "Conspiratorial secrecy"),
    (r"100%\s*(proven|confirmed|guaranteed)",            "False certainty claims"),
    (r"[A-Z]{4,}",                                       "Excessive capitalization"),
    (r"!!!+",                                            "Excessive exclamation marks"),
    # Indian fake news patterns
    (r"(free|muft).{0,20}(laptop|mobile|phone|tab)",    "Free gadget scam"),
    (r"only \d+ slots? (remaining|left)",                "False scarcity claim"),
    (r"apply (immediately|now|before).{0,20}(deadline|expires|forever)", "Urgency manipulation"),
    (r"(anonymous|secret).{0,20}(source|insider|officer)", "Anonymous source claim"),
    (r"opposition media (is|are) hiding",                "Media conspiracy claim"),
    (r"(government|sarkar).{0,20}secretly",             "Secret government claim"),
    (r"share with every",                                "Viral sharing manipulation"),
    (r"applications? close.{0,20}forever",               "False deadline urgency"),
]

# High-risk patterns for override logic
HIGH_RISK_PATTERNS = {
    "False scarcity claim",
    "Urgency manipulation",
    "Anonymous source claim",
    "False deadline urgency",
    "Viral sharing manipulation",
    "Free gadget scam",
    "Secret government claim",
    "Media conspiracy claim",
}


def find_suspicious_patterns(text: str) -> list[str]:
    """Find suspicious patterns — deduplicated results."""
    found = []
    seen  = set()
    for pattern, label in SUSPICIOUS_PATTERNS:
        if label not in seen and re.search(pattern, text, re.IGNORECASE):
            found.append(label)
            seen.add(label)
    return found


def get_influential_words(text: str, top_n: int = 15) -> list[WordInfluence]:
    """Extract top words influencing prediction using LR coefficients."""
    if tfidf_vectorizer is None or feature_names is None or all_models is None:
        return []
    try:
        lr_model = all_models.get("Logistic Regression")
        if lr_model is None:
            return []
        coefficients = lr_model.coef_[0]
        processed    = preprocess_text(text)
        vec          = tfidf_vectorizer.transform([processed])
        dense        = vec.toarray()[0]
        nonzero_idx  = np.nonzero(dense)[0]
        if len(nonzero_idx) == 0:
            return []
        scores     = dense[nonzero_idx] * coefficients[nonzero_idx]
        words      = [feature_names[i] for i in nonzero_idx]
        word_scores = sorted(zip(words, scores),
                             key=lambda x: abs(x[1]), reverse=True)[:top_n]
        return [
            WordInfluence(
                word=w,
                score=round(float(abs(s)), 4),
                direction="fake" if s < 0 else "real",
            )
            for w, s in word_scores
        ]
    except Exception as e:
        logger.error(f"Explainability error: {e}")
        return []


def build_explanation(
    prediction: str,
    confidence: float,
    influential_words: list[WordInfluence],
    suspicious_patterns: list[str],
) -> str:
    """Generate a human-readable explanation."""
    conf_pct = round(confidence * 100, 1)

    if prediction == "Fake":
        if conf_pct > 85:
            base = f"This article shows strong indicators of misinformation (confidence: {conf_pct}%)."
        elif conf_pct > 65:
            base = f"This article has several characteristics common in fake news (confidence: {conf_pct}%)."
        else:
            base = f"This article leans toward fake news but with moderate confidence ({conf_pct}%)."
    else:
        if conf_pct > 85:
            base = f"This article exhibits strong markers of credible journalism (confidence: {conf_pct}%)."
        elif conf_pct > 65:
            base = f"This article has characteristics consistent with factual reporting (confidence: {conf_pct}%)."
        else:
            base = f"This article leans toward being real news, though with moderate confidence ({conf_pct}%)."

    fake_words = [w.word for w in influential_words if w.direction == "fake"][:3]
    real_words = [w.word for w in influential_words if w.direction == "real"][:3]

    if fake_words:
        base += f" Terms like '{', '.join(fake_words)}' are associated with misleading content."
    if real_words:
        base += f" Terms like '{', '.join(real_words)}' are associated with factual reporting."
    if suspicious_patterns:
        base += f" Detected suspicious patterns: {'; '.join(suspicious_patterns[:3])}."

    return base


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Fake News Detector API v2.0", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if all_models is not None else "models_not_loaded",
        "models_loaded": list(all_models.keys()) if all_models else [],
        "model_results": model_results or {},
        "num_features": len(feature_names) if feature_names is not None else 0,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if all_models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0   = time.perf_counter()
    text = req.text

    # Preprocess & vectorize
    processed = preprocess_text(text)
    if not processed.strip():
        raise HTTPException(status_code=422,
                            detail="Text too short after preprocessing.")

    vec = tfidf_vectorizer.transform([processed])

    # ── Get predictions from ALL 3 models ─────────────────────────────────────
    model_predictions: dict[str, ModelResult] = {}
    for name, model in all_models.items():
        proba = model.predict_proba(vec)[0]
        model_predictions[name] = ModelResult(
            fake_probability=round(float(proba[0]), 4),
            real_probability=round(float(proba[1]), 4),
            prediction="Real" if proba[1] > proba[0] else "Fake",
        )

    # ── Primary verdict — SVM (best accuracy) ─────────────────────────────────
    primary    = model_predictions.get("SVM") or list(model_predictions.values())[0]
    fake_prob  = primary.fake_probability
    real_prob  = primary.real_probability
    prediction = primary.prediction
    confidence = max(fake_prob, real_prob)

    # ── Explainability ─────────────────────────────────────────────────────────
    influential_words  = get_influential_words(text)
    suspicious_patterns = find_suspicious_patterns(text)

    # ── Pattern-based override ─────────────────────────────────────────────────
    # If 2+ high-risk patterns detected AND model is weakly "Real" → override to Fake
    high_risk_count = sum(1 for p in suspicious_patterns if p in HIGH_RISK_PATTERNS)
    if high_risk_count >= 2 and prediction == "Real" and confidence < 0.80:
        prediction = "Fake"
        fake_prob  = 0.75
        real_prob  = 0.25
        confidence = 0.75
        logger.info(f"Pattern override triggered: {high_risk_count} high-risk patterns found")

    explanation = build_explanation(prediction, confidence, influential_words, suspicious_patterns)
    elapsed     = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        fake_probability=round(fake_prob, 4),
        real_probability=round(real_prob, 4),
        influential_words=influential_words,
        explanation=explanation,
        processing_time_ms=round(elapsed, 2),
        word_count=len(text.split()),
        suspicious_patterns=suspicious_patterns,
        model_predictions=model_predictions,
    )