"""
Chargement donné + modèles en cache
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATA_CLEAN, DATA_PROCESSED,
    TFIDF_LR_PATH, TFIDF_SVM_PATH, BERT_DIR,
    TEXT_COL, CLEAN_COL, RATING_COL, INSURER_COL, SENTIMENT_COL,
)


@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Charge le dataset nettoyé."""
    candidates = [
        DATA_PROCESSED / "reviews_topics.csv",
        DATA_CLEAN,
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(columns={
                TEXT_COL: "review",
                CLEAN_COL: "review_clean",
                RATING_COL: "stars",
                INSURER_COL: "insurer",
                SENTIMENT_COL: "sentiment",
            })
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
        (TFIDF_SVM_PATH, "TF-IDF + SVM"),
        (TFIDF_LR_PATH, "TF-IDF + Logistic Regression"),
    ]
    for path, name in candidates:
        if path.exists():
            return joblib.load(path), name
    return None, None


@st.cache_resource
def load_bert():
    """Charge le modèle BERT fine-tuné si disponible."""
    if not BERT_DIR.exists():
        return None
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "text-classification",
            model=str(BERT_DIR),
            tokenizer=str(BERT_DIR),
            top_k=None,
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
            model="facebook/bart-large-mnli",
        )
    except Exception:
        return None
