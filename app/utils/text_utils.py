"""
Utilitaires de traitement du texte.
"""

import re

import pandas as pd


def preprocess_text(text: str) -> str:
    """Nettoyage minimal pour la prédiction."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # TODO upgrade
    return text


def build_tfidf_index(df: pd.DataFrame):
    """Construit l'index TF-IDF pour la recherche IR."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = df["review_clean"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix
