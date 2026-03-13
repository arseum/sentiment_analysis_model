"""
Chargement donné + modèles en cache
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Charge le dataset nettoyé."""
    candidates = [
        DATA_PROCESSED / "reviews_topics.csv",
        DATA_PROCESSED / "reviews_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            col_map = {}
            for col in df.columns:
                if col.lower() in ("review", "review_text", "text", "content"):
                    col_map[col] = "review"
                elif col.lower() in ("review_clean", "clean_text", "cleaned"):
                    col_map[col] = "review_clean"
                elif col.lower() in ("stars", "rating", "note", "score"):
                    col_map[col] = "stars"
                elif col.lower() in ("insurer", "company", "company_name", "assureur"):
                    col_map[col] = "insurer"
            df = df.rename(columns=col_map)
            if "review_clean" not in df.columns and "review" in df.columns:
                df["review_clean"] = df["review"]
            if "sentiment" not in df.columns and "stars" in df.columns:
                df["sentiment"] = df["stars"].apply(
                    lambda s: "négatif" if s <= 2 else ("neutre" if s == 3 else "positif")
                )
            return df
    return None


@st.cache_resource
def load_model():
    """Charge le meilleur modèle classique disponible."""
    candidates = [
        (MODELS_DIR / "tfidf_svm.pkl", "TF-IDF + SVM"),
        (MODELS_DIR / "tfidf_lr.pkl", "TF-IDF + Logistic Regression"),
    ]
    for path, name in candidates:
        if path.exists():
            return joblib.load(path), name
    return None, None


@st.cache_resource
def load_bert():
    """Charge le modèle BERT fine-tuné si disponible."""
    bert_dir = MODELS_DIR / "bert_sentiment"
    if not bert_dir.exists():
        return None
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "text-classification",
            model=str(bert_dir),
            tokenizer=str(bert_dir),
            return_all_scores=True,
        )
    except Exception:
        return None


@st.cache_resource
def load_zero_shot():
    """Charge le pipeline zero-shot pour la catégorisation."""
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
        )
    except Exception:
        return None
