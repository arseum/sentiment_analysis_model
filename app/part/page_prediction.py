"""
Page 1 — Prédiction de sentiment & catégorie.
"""

import pandas as pd
import streamlit as st

from utils.loaders import load_bert, load_zero_shot
from utils.text_utils import preprocess_text

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ZS_CATEGORY_LABELS


def _predict_sentiment(user_input: str, model, model_name: str):
    """Retourne (pred_label, scores dict, used_model str)."""
    bert = load_bert()
    if bert is not None:
        with st.spinner("Analyse BERT..."):
            results = bert(user_input[:512])
            # top_k=None renvoie une liste plate [{"label":…, "score":…}, …]
            items = results[0] if isinstance(results[0], list) else results
            label_map = {
                "label_0": "négatif",
                "label_1": "neutre",
                "label_2": "positif",
                "negative": "négatif",
                "neutral": "neutre",
                "positive": "positif",
            }
            scores = {
                label_map.get(r["label"].lower(), r["label"]): r["score"]
                for r in items
            }
            pred_label = max(scores, key=scores.get)
            return pred_label, scores, "BERT fine-tuné"

    if model is not None:
        import numpy as np
        id2label = {0: "négatif", 1: "neutre", 2: "positif"}
        clean_input = preprocess_text(user_input)
        raw_pred = model.predict([clean_input])[0]
        pred_label = id2label.get(int(raw_pred), str(raw_pred))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([clean_input])[0]
            classes = [id2label.get(int(c), str(c)) for c in model.classes_]
            scores = dict(zip(classes, proba))
        else:
            scores = {pred_label: 1.0}
        return pred_label, scores, model_name

    return None, None, None


def _show_sentiment(user_input: str, model, model_name: str):
    st.subheader("Sentiment prédit")
    pred_label, scores, used_model = _predict_sentiment(user_input, model, model_name)
    if pred_label is None:
        st.error("Aucun modèle disponible. Veuillez d'abord exécuter le notebook 5.")
        st.stop()

    st.info(f"Modèle utilisé : **{used_model}**")

    col_a, col_b, col_c = st.columns(3)
    for col, (lbl, scr) in zip([col_a, col_b, col_c], scores.items()):
        with col:
            st.metric(
                label=lbl.capitalize(),
                value=f"{scr:.1%}",
                delta="prédit" if lbl == pred_label else "",
            )
    st.success(f"Sentiment : **{pred_label.upper()}**")


def _show_category(user_input: str):
    st.subheader("Catégorie détectée")
    zs = load_zero_shot()
    if zs is None:
        st.warning(
            "Pipeline zero-shot non disponible. Installez `transformers` et `torch`."
        )
        return

    import plotly.express as px

    st.info("Modèle utilisé : **Zero-shot NLI (cross-encoder/nli-MiniLM2-L6-H768)**")

    with st.spinner("Classification zero-shot..."):
        cat_result = zs(user_input[:512], candidate_labels=ZS_CATEGORY_LABELS)

    top_cat = cat_result["labels"][0]
    top_score = cat_result["scores"][0]

    cat_df = pd.DataFrame(
        {
            "Catégorie": [l.capitalize() for l in cat_result["labels"]],
            "Score": cat_result["scores"],
        }
    )
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
    st.info(f"Catégorie principale : **{top_cat.capitalize()}** ({top_score:.1%})")


def render(model, model_name: str):
    st.title("Prédiction de sentiment & catégorie")
    st.markdown(
        "Saisissez un avis d'assurance pour obtenir le **sentiment prédit** "
        "et la **catégorie détectée** automatiquement."
    )

    user_input = st.text_area(
        "Avis à analyser",
        height=150,
        placeholder="Ex: Mon sinistre auto a été traité rapidement, très satisfait du service.",
    )

    col1, _ = st.columns([1, 3])
    with col1:
        predict_btn = st.button("Analyser", type="primary", use_container_width=True)

    if predict_btn and user_input.strip():
        _show_sentiment(user_input, model, model_name)
        _show_category(user_input)
