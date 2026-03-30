"""
Page 4 — Recherche de reviews similaires (TF-IDF / IR).
"""

import re

import pandas as pd
import streamlit as st

from utils.text_utils import preprocess_text


def _apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Affiche les filtres et retourne le dataframe filtré + top_k."""
    with st.expander("Filtres avancés", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "insurer" in df.columns:
                insurers_filter = ["Tous"] + sorted(df["insurer"].dropna().unique().tolist())
                selected_insurer = st.selectbox("Assureur", insurers_filter)
            else:
                selected_insurer = "Tous"

        with col2:
            if "stars" in df.columns:
                min_stars, max_stars = st.slider("Étoiles", 1, 5, (1, 5))
            else:
                min_stars, max_stars = 1, 5

        with col3:
            if "sentiment" in df.columns:
                sentiment_filter = st.multiselect(
                    "Sentiment",
                    ["positif", "neutre", "négatif"],
                    default=["positif", "neutre", "négatif"],
                )
            else:
                sentiment_filter = []

        with col4:
            top_k = st.slider("Nombre de résultats", 3, 20, 5)

    df_filtered = df.copy()
    if selected_insurer != "Tous" and "insurer" in df.columns:
        df_filtered = df_filtered[df_filtered["insurer"] == selected_insurer]
    if "stars" in df.columns:
        df_filtered = df_filtered[
            (df_filtered["stars"] >= min_stars) & (df_filtered["stars"] <= max_stars)
        ]
    if sentiment_filter and "sentiment" in df.columns:
        df_filtered = df_filtered[df_filtered["sentiment"].isin(sentiment_filter)]

    return df_filtered, top_k


def _search(query: str, df_filtered: pd.DataFrame, top_k: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    text_col = "review_clean" if "review_clean" in df_filtered.columns else "review"
    corpus = df_filtered[text_col].fillna("").tolist()

    with st.spinner("Calcul des similarités..."):
        vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([preprocess_text(query)])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[::-1][:top_k]
    display_col = "review" if "review" in df_filtered.columns else text_col
    query_terms = set(preprocess_text(query).split())

    st.subheader(f"Top {top_k} résultats")
    for rank, idx in enumerate(top_indices, 1):
        row = df_filtered.iloc[idx]
        score = similarities[idx]

        if score < 0.01:
            st.warning("Pas de résultats suffisamment similaires.")
            break

        stars_str = f"{int(row.get('stars', '?'))} étoiles | " if "stars" in row else ""
        insurer_str = f"{row.get('insurer', '')} | " if "insurer" in row else ""
        sentiment_str = str(row.get("sentiment", "")).capitalize() if "sentiment" in row else ""

        with st.expander(
            f"#{rank} — Score : {score:.3f} | {stars_str}{insurer_str}{sentiment_str}",
            expanded=(rank == 1),
        ):
            review_text = str(row.get(display_col, ""))
            st.write(review_text)

            highlighted = review_text
            for term in query_terms:
                if len(term) > 2:
                    highlighted = re.sub(
                        f"(?i)\\b{re.escape(term)}\\b", f"**{term}**", highlighted
                    )
            if highlighted != review_text:
                st.markdown("**Termes trouvés :**")
                st.markdown(highlighted[:500])

            meta_cols = st.columns(4)
            with meta_cols[0]:
                st.metric("Similarité", f"{score:.3f}")
            if "stars" in row:
                with meta_cols[1]:
                    st.metric("Étoiles", f"{int(row['stars'])} / 5")
            if "insurer" in row:
                with meta_cols[2]:
                    st.metric("Assureur", str(row["insurer"]))
            if "sentiment" in row:
                with meta_cols[3]:
                    st.metric("Sentiment", str(row["sentiment"]).capitalize())


def render(df: pd.DataFrame):
    st.title("Recherche de reviews similaires (TF-IDF)")
    st.markdown(
        "Recherche par similarité cosine TF-IDF — "
        "réutilisation du pipeline du **Projet 1**."
    )

    if df is None:
        st.error("Dataset non chargé. Exécutez les notebooks 1 et 2.")
        st.stop()

    df_filtered, top_k = _apply_filters(df)
    st.info(f"**{len(df_filtered)} reviews** correspondent aux filtres.")

    query = st.text_input(
        "Requête de recherche",
        placeholder="Ex: sinistre auto remboursement refusé",
    )

    if st.button("Rechercher", type="primary") and query.strip():
        if len(df_filtered) == 0:
            st.warning("Aucune review avec ces filtres.")
        else:
            _search(query, df_filtered, top_k)
