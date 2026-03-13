"""
Page 2 — Résumé par assureur.
"""

import pandas as pd
import streamlit as st


def _show_metrics(sub: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    if "stars" in sub.columns:
        with col1:
            st.metric("Note moyenne", f"{sub['stars'].mean():.2f} / 5")
        with col2:
            st.metric("Médiane", f"{sub['stars'].median():.1f} / 5")
    if "sentiment" in sub.columns:
        counts = sub["sentiment"].value_counts()
        with col3:
            pct_pos = counts.get("positif", 0) / len(sub) * 100
            st.metric("% Positif", f"{pct_pos:.1f}%")
        with col4:
            pct_neg = counts.get("négatif", 0) / len(sub) * 100
            st.metric("% Négatif", f"{pct_neg:.1f}%")


def _show_charts(sub: pd.DataFrame):
    import plotly.express as px

    col_left, col_right = st.columns(2)
    if "sentiment" in sub.columns:
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


def _show_summary(sub: pd.DataFrame):
    st.subheader("Résumé automatique des reviews")
    n = st.slider("Nombre de reviews à résumer", min_value=5, max_value=50, value=20)

    if not st.button("Générer le résumé", type="primary"):
        return

    text_col = "review_clean" if "review_clean" in sub.columns else "review"
    combined_text = " ".join(sub[text_col].dropna().head(n).tolist())[:3000]

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
        st.markdown("**Résumé automatique :**")
        st.write(summary)
    except Exception:
        sentences = combined_text.split(". ")
        st.markdown("**Résumé extractif (fallback) :**")
        st.write(". ".join(sentences[:5]) + ".")


def _show_examples(sub: pd.DataFrame):
    st.subheader("Exemples de reviews")
    text_col = "review" if "review" in sub.columns else "review_clean"
    tab_pos, tab_neu, tab_neg = st.tabs(["Positives", "Neutres", "Négatives"])
    for tab, sent in zip([tab_pos, tab_neu, tab_neg], ["positif", "neutre", "négatif"]):
        with tab:
            examples = sub[sub["sentiment"] == sent][text_col].dropna().head(3)
            for i, txt in enumerate(examples, 1):
                st.markdown(f"**{i}.** {txt[:300]}...")


def render(df: pd.DataFrame):
    st.title("Résumé par assureur")

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

    _show_metrics(sub)
    if "sentiment" in sub.columns:
        _show_charts(sub)
    _show_summary(sub)
    _show_examples(sub)
