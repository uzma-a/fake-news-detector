"""
Multi-Model Fake News Detector
Trains and compares LR, Naive Bayes, and SVM
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def train_all_models(data_dir: str = "."):
    # Load dataset
    combined = os.path.join(data_dir, "combined_dataset.csv")
    fake_csv = os.path.join(data_dir, "Fake.csv")
    true_csv = os.path.join(data_dir, "True.csv")

    if os.path.exists(combined):
        print("Loading combined dataset...")
        df = pd.read_csv(combined)
    elif os.path.exists(fake_csv):
        print("Loading Kaggle dataset...")
        fake = pd.read_csv(fake_csv)
        real = pd.read_csv(true_csv)
        fake["label"] = 0
        real["label"] = 1
        fake["content"] = fake.get("title", pd.Series()).fillna("") + " " + fake.get("text", pd.Series()).fillna("")
        real["content"] = real.get("title", pd.Series()).fillna("") + " " + real.get("text", pd.Series()).fillna("")
        df = pd.concat([fake[["content","label"]], real[["content","label"]]])
    else:
        raise FileNotFoundError("No dataset found!")

    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""]

    print(f"Dataset: {len(df)} samples")
    print("Preprocessing...")
    df["processed"] = df["content"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["processed"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF — shared vectorizer
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # Define 3 models
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight="balanced", random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=1000,
                      class_weight="balanced", random_state=42)
        ),
    }

    results = {}
    trained_models = {}

    print("\n=== Training All Models ===\n")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": round(float(acc), 4),
            "model_type": name,
        }
        trained_models[name] = model
        print(f"✅ {name}: {acc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=["Fake","Real"]))

    # Save everything
    print("\n=== Saving Models ===")
    joblib.dump(tfidf,           os.path.join(data_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(trained_models,  os.path.join(data_dir, "all_models.joblib"))
    joblib.dump(results,         os.path.join(data_dir, "model_results.joblib"))
    feature_names = tfidf.get_feature_names_out()
    joblib.dump(feature_names,   os.path.join(data_dir, "feature_names.joblib"))

    print("✅ All models saved!")
    print("\n=== Final Comparison ===")
    for name, res in results.items():
        print(f"{name:25} → {res['accuracy']*100:.2f}%")

    return tfidf, trained_models, results


if __name__ == "__main__":
    train_all_models(data_dir=".")