"""
Application Streamlit - Analyse d'avis d'assurance
Projet 2 NLP 2026 - Arsène Maitre & Gabriel Thibout

4 pages :
  1. Prédiction de sentiment + catégorie
  2. Résumé par assureur
  3. Explication LIME
  4. Recherche IR (TF-IDF + cosine)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse d'avis d'assurance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Charge le dataset nettoyé."""
    candidates = [
        DATA_PROCESSED / "reviews_topics.csv",
        DATA_PROCESSED / "reviews_clean.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            # Normalise les noms de colonnes
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
    """Charge le meilleur modèle disponible."""
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
            model="cross-encoder/nli-MiniLM2-L6-H768",  # léger
        )
    except Exception:
        return None


def preprocess_text(text: str) -> str:
    """Nettoyage minimal pour la prédiction."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf_index(df: pd.DataFrame):
    """Construit l'index TF-IDF pour la recherche IR."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = df["review_clean"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ Avis d'assurance")
st.sidebar.markdown("**NLP Projet 2 — 2026**")
st.sidebar.markdown("Arsène Maitre & Gabriel Thibout")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Prédiction",
        "📊 Résumé par assureur",
        "🔍 Explication LIME",
        "🔎 Recherche IR",
    ],
)

# ── Chargement global ─────────────────────────────────────────────────────────

df = load_data()
model, model_name = load_model()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔮 Prédiction":
    st.title("🔮 Prédiction de sentiment & catégorie")
    st.markdown(
        "Saisissez une review d'assurance pour obtenir le **sentiment prédit** "
        "et la **catégorie détectée** automatiquement."
    )

    user_input = st.text_area(
        "Review à analyser",
        height=150,
        placeholder="Ex: The insurance company has been very helpful with my claim...",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        predict_btn = st.button("Analyser", type="primary", use_container_width=True)

    if predict_btn and user_input.strip():
        clean_input = preprocess_text(user_input)

        # ── Sentiment ──────────────────────────────────────────────────────────
        st.subheader("Sentiment prédit")

        bert = load_bert()
        if bert is not None:
            with st.spinner("Analyse BERT..."):
                results = bert(user_input[:512])
                # Résultats sous forme [{label, score}, ...]
                label_map = {
                    "LABEL_0": "négatif",
                    "LABEL_1": "neutre",
                    "LABEL_2": "positif",
                    "negative": "négatif",
                    "neutral": "neutre",
                    "positive": "positif",
                }
                scores = {
                    label_map.get(r["label"].lower(), r["label"]): r["score"]
                    for r in results[0]
                }
                pred_label = max(scores, key=scores.get)
                used_model = "BERT fine-tuné"
        elif model is not None:
            id2label = {0: "négatif", 1: "neutre", 2: "positif"}
            raw_pred = model.predict([clean_input])[0]
            pred_label = id2label.get(int(raw_pred), str(raw_pred))
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([clean_input])[0]
                classes = [id2label.get(int(c), str(c)) for c in model.classes_]
                scores = dict(zip(classes, proba))
            else:
                scores = {pred_label: 1.0}
            used_model = model_name
        else:
            st.error("Aucun modèle disponible. Veuillez d'abord exécuter le notebook 5.")
            st.stop()

        color_map = {"positif": "🟢", "neutre": "🟡", "négatif": "🔴"}
        icon = color_map.get(pred_label, "⚪")

        col_a, col_b, col_c = st.columns(3)
        for col, (lbl, scr) in zip(
            [col_a, col_b, col_c],
            scores.items(),
        ):
            with col:
                delta = "✓ prédit" if lbl == pred_label else ""
                st.metric(
                    label=f"{color_map.get(lbl, '')} {lbl.capitalize()}",
                    value=f"{scr:.1%}",
                    delta=delta,
                )

        st.success(f"{icon} **Sentiment : {pred_label.upper()}** (modèle : {used_model})")

        # ── Catégorie zero-shot ────────────────────────────────────────────────
        st.subheader("Catégorie détectée")
        zs = load_zero_shot()
        if zs is not None:
            with st.spinner("Classification zero-shot..."):
                category_labels = [
                    "pricing and cost",
                    "coverage and benefits",
                    "enrollment process",
                    "customer service",
                    "claims processing",
                    "policy cancellation",
                ]
                cat_result = zs(user_input[:512], candidate_labels=category_labels)
                label_fr = {
                    "pricing and cost": "Tarification",
                    "coverage and benefits": "Couverture",
                    "enrollment process": "Souscription",
                    "customer service": "Service client",
                    "claims processing": "Remboursements",
                    "policy cancellation": "Résiliation",
                }
                top_cat = cat_result["labels"][0]
                top_score = cat_result["scores"][0]

                cat_df = pd.DataFrame(
                    {
                        "Catégorie": [
                            label_fr.get(l, l) for l in cat_result["labels"]
                        ],
                        "Score": cat_result["scores"],
                    }
                )
                import plotly.express as px

                fig = px.bar(
                    cat_df,
                    x="Score",
                    y="Catégorie",
                    orientation="h",
                    color="Score",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    f"📂 Catégorie principale : **{label_fr.get(top_cat, top_cat)}** ({top_score:.1%})"
                )
        else:
            st.warning(
                "Pipeline zero-shot non disponible. "
                "Installez `transformers` et `torch`."
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : RÉSUMÉ PAR ASSUREUR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Résumé par assureur":
    st.title("📊 Résumé par assureur")

    if df is None:
        st.error("Dataset non chargé. Exécutez les notebooks 1 et 2 d'abord.")
        st.stop()

    if "insurer" not in df.columns:
        st.error("Colonne 'insurer' introuvable dans le dataset.")
        st.stop()

    insurers = sorted(df["insurer"].dropna().unique().tolist())
    selected = st.selectbox("Choisir un assureur", insurers)

    sub = df[df["insurer"] == selected].copy()
    st.markdown(f"**{len(sub)} reviews** pour {selected}")

    # ── Métriques clés ─────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    if "stars" in sub.columns:
        with col1:
            st.metric("Note moyenne", f"{sub['stars'].mean():.2f} ⭐")
        with col2:
            st.metric("Médianne", f"{sub['stars'].median():.1f} ⭐")
    if "sentiment" in sub.columns:
        sentiment_counts = sub["sentiment"].value_counts()
        with col3:
            pct_pos = sentiment_counts.get("positif", 0) / len(sub) * 100
            st.metric("% Positif", f"{pct_pos:.1f}%")
        with col4:
            pct_neg = sentiment_counts.get("négatif", 0) / len(sub) * 100
            st.metric("% Négatif", f"{pct_neg:.1f}%")

    # ── Distribution sentiment ─────────────────────────────────────────────────
    if "sentiment" in sub.columns:
        import plotly.express as px

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Distribution des sentiments")
            fig_sent = px.bar(
                sub["sentiment"].value_counts().reset_index(),
                x="sentiment",
                y="count",
                color="sentiment",
                color_discrete_map={
                    "positif": "#2ecc71",
                    "neutre": "#f39c12",
                    "négatif": "#e74c3c",
                },
            )
            fig_sent.update_layout(showlegend=False)
            st.plotly_chart(fig_sent, use_container_width=True)

        if "stars" in sub.columns:
            with col_right:
                st.subheader("Distribution des étoiles")
                fig_stars = px.histogram(
                    sub, x="stars", nbins=5, color_discrete_sequence=["#3498db"]
                )
                st.plotly_chart(fig_stars, use_container_width=True)

    # ── Résumé automatique ─────────────────────────────────────────────────────
    st.subheader("Résumé automatique des reviews")
    n_reviews_summary = st.slider(
        "Nombre de reviews à résumer", min_value=5, max_value=50, value=20
    )

    if st.button("Générer le résumé", type="primary"):
        text_col = "review_clean" if "review_clean" in sub.columns else "review"
        reviews_sample = sub[text_col].dropna().head(n_reviews_summary).tolist()
        combined_text = " ".join(reviews_sample)[:3000]  # Limit to 3000 chars

        try:
            from transformers import pipeline as hf_pipeline

            with st.spinner("Résumé en cours (T5/BART)..."):
                summarizer = hf_pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    max_length=150,
                    min_length=50,
                    truncation=True,
                )
                summary = summarizer(combined_text)[0]["summary_text"]
                st.success("**Résumé automatique :**")
                st.write(summary)
        except Exception as e:
            # Fallback: extractive summary (top sentences)
            sentences = combined_text.split(". ")
            extractive = ". ".join(sentences[:5]) + "."
            st.info("**Résumé extractif (fallback) :**")
            st.write(extractive)

    # ── Exemples de reviews ────────────────────────────────────────────────────
    st.subheader("Exemples de reviews")
    text_col = "review" if "review" in sub.columns else "review_clean"
    tab_pos, tab_neu, tab_neg = st.tabs(["✅ Positives", "😐 Neutres", "❌ Négatives"])
    for tab, sent in zip([tab_pos, tab_neu, tab_neg], ["positif", "neutre", "négatif"]):
        with tab:
            examples = sub[sub["sentiment"] == sent][text_col].dropna().head(3)
            for i, txt in enumerate(examples, 1):
                st.markdown(f"**{i}.** {txt[:300]}...")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : EXPLICATION LIME
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Explication LIME":
    st.title("🔍 Explication des prédictions (LIME)")
    st.markdown(
        "LIME explique pourquoi le modèle a prédit un certain sentiment en "
        "identifiant les mots les plus influents."
    )

    if model is None:
        st.error("Aucun modèle disponible. Exécutez le notebook 5.")
        st.stop()

    user_input = st.text_area(
        "Review à expliquer",
        height=150,
        value="The claim process was a nightmare. They denied my request without explanation and the customer service was rude.",
    )

    num_features = st.slider("Nombre de mots à expliquer", 5, 20, 10)

    if st.button("Expliquer avec LIME", type="primary"):
        try:
            from lime.lime_text import LimeTextExplainer

            class_names = ["négatif", "neutre", "positif"]

            # Adapter predict_proba selon le modèle
            if hasattr(model, "predict_proba"):
                predict_fn = lambda texts: model.predict_proba(
                    [preprocess_text(t) for t in texts]
                )
            else:
                # LinearSVC: utiliser decision_function + softmax
                def predict_fn(texts):
                    cleaned = [preprocess_text(t) for t in texts]
                    decision = model.decision_function(cleaned)
                    # Softmax approximation
                    exp = np.exp(decision - decision.max(axis=1, keepdims=True))
                    return exp / exp.sum(axis=1, keepdims=True)

            explainer = LimeTextExplainer(class_names=class_names, random_state=42)
            with st.spinner("Calcul LIME en cours..."):
                exp = explainer.explain_instance(
                    user_input,
                    predict_fn,
                    num_features=num_features,
                    num_samples=500,
                )

            # ── Scores de confiance ─────────────────────────────────────────
            st.subheader("Scores de confiance")
            proba = predict_fn([user_input])[0]
            col1, col2, col3 = st.columns(3)
            emojis = ["🔴", "🟡", "🟢"]
            for col, (cls, prob, em) in zip(
                [col1, col2, col3], zip(class_names, proba, emojis)
            ):
                with col:
                    st.metric(f"{em} {cls.capitalize()}", f"{prob:.1%}")

            pred_idx = np.argmax(proba)
            st.success(
                f"**Prédiction : {class_names[pred_idx].upper()}** "
                f"({proba[pred_idx]:.1%} de confiance)"
            )

            # ── Explication LIME ────────────────────────────────────────────
            st.subheader("Mots les plus influents")

            # HTML de LIME
            lime_html = exp.as_html()
            st.components.v1.html(lime_html, height=400, scrolling=True)

            # Tableau des contributions
            st.subheader("Contributions par mot")
            lime_list = exp.as_list()
            lime_df = pd.DataFrame(lime_list, columns=["Mot", "Contribution"])
            lime_df["Impact"] = lime_df["Contribution"].apply(
                lambda x: "✅ Pro-positif" if x > 0 else "❌ Pro-négatif"
            )
            lime_df["Contribution"] = lime_df["Contribution"].round(4)
            lime_df = lime_df.sort_values("Contribution", ascending=False)
            st.dataframe(
                lime_df.style.background_gradient(
                    subset=["Contribution"], cmap="RdYlGn", vmin=-0.5, vmax=0.5
                ),
                use_container_width=True,
            )

        except ImportError:
            st.error("LIME non installé. Exécutez : `pip install lime`")
        except Exception as e:
            st.error(f"Erreur LIME : {e}")
            st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 : RECHERCHE IR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Recherche IR":
    st.title("🔎 Recherche de reviews similaires (TF-IDF)")
    st.markdown(
        "Recherche basée sur la similarité cosine TF-IDF — "
        "réutilisation du pipeline du **Projet 1**."
    )

    if df is None:
        st.error("Dataset non chargé. Exécutez les notebooks 1 et 2.")
        st.stop()

    # ── Filtres ────────────────────────────────────────────────────────────────
    with st.expander("🔧 Filtres avancés", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "insurer" in df.columns:
                insurers_filter = ["Tous"] + sorted(
                    df["insurer"].dropna().unique().tolist()
                )
                selected_insurer = st.selectbox("Assureur", insurers_filter)
            else:
                selected_insurer = "Tous"

        with col2:
            if "stars" in df.columns:
                min_stars, max_stars = st.slider(
                    "Étoiles", min_value=1, max_value=5, value=(1, 5)
                )
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

    # Appliquer les filtres
    df_filtered = df.copy()
    if selected_insurer != "Tous" and "insurer" in df.columns:
        df_filtered = df_filtered[df_filtered["insurer"] == selected_insurer]
    if "stars" in df.columns:
        df_filtered = df_filtered[
            (df_filtered["stars"] >= min_stars) & (df_filtered["stars"] <= max_stars)
        ]
    if sentiment_filter and "sentiment" in df.columns:
        df_filtered = df_filtered[df_filtered["sentiment"].isin(sentiment_filter)]

    st.info(f"**{len(df_filtered)} reviews** correspondent aux filtres.")

    # ── Requête ────────────────────────────────────────────────────────────────
    query = st.text_input(
        "Votre requête de recherche",
        placeholder="Ex: denied claim unfair treatment",
    )

    if st.button("Rechercher", type="primary") and query.strip():
        if len(df_filtered) == 0:
            st.warning("Aucune review avec ces filtres.")
        else:
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

            st.subheader(f"Top {top_k} résultats")

            display_text_col = "review" if "review" in df_filtered.columns else text_col

            for rank, idx in enumerate(top_indices, 1):
                row = df_filtered.iloc[idx]
                score = similarities[idx]

                if score < 0.01:
                    st.warning("Pas de résultats suffisamment similaires.")
                    break

                with st.expander(
                    f"#{rank} — Score: {score:.3f} | "
                    + (f"⭐ {int(row.get('stars', '?'))} | " if "stars" in row else "")
                    + (f"{row.get('insurer', '')} | " if "insurer" in row else "")
                    + (row.get("sentiment", "").capitalize() if "sentiment" in row else ""),
                    expanded=(rank == 1),
                ):
                    review_text = row.get(display_text_col, "")
                    st.write(review_text)

                    # Highlight query terms
                    query_terms = set(preprocess_text(query).split())
                    highlighted = review_text
                    for term in query_terms:
                        if len(term) > 2:
                            highlighted = re.sub(
                                f"(?i)\\b{re.escape(term)}\\b",
                                f"**{term}**",
                                highlighted,
                            )

                    if highlighted != review_text:
                        st.markdown("**Termes trouvés :**")
                        st.markdown(highlighted[:500])

                    # Métadonnées
                    meta_cols = st.columns(4)
                    with meta_cols[0]:
                        st.metric("Similarité", f"{score:.3f}")
                    if "stars" in row:
                        with meta_cols[1]:
                            st.metric("Étoiles", f"{'⭐' * int(row['stars'])}")
                    if "insurer" in row:
                        with meta_cols[2]:
                            st.metric("Assureur", str(row["insurer"]))
                    if "sentiment" in row:
                        with meta_cols[3]:
                            em = {"positif": "🟢", "neutre": "🟡", "négatif": "🔴"}
                            st.metric(
                                "Sentiment",
                                f"{em.get(row['sentiment'], '')} {str(row['sentiment']).capitalize()}",
                            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(
    """
**Guide rapide :**
1. 🔮 Analyser une review
2. 📊 Explorer un assureur
3. 🔍 Comprendre les prédictions
4. 🔎 Rechercher des reviews

**Avant utilisation :**
```bash
# Exécuter les notebooks 1-5
# puis :
streamlit run app/streamlit_app.py
```
"""
)
