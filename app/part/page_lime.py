"""
Page 3 — Explication des prédictions avec LIME.
"""

import numpy as np
import pandas as pd
import streamlit as st

from utils.text_utils import preprocess_text

CLASS_NAMES = ["négatif", "neutre", "positif"]


def _build_predict_fn(model):
    """Construit la fonction de prédiction compatible LIME."""
    if hasattr(model, "predict_proba"):
        return lambda texts: model.predict_proba([preprocess_text(t) for t in texts])

    def predict_fn(texts):
        cleaned = [preprocess_text(t) for t in texts]
        decision = model.decision_function(cleaned)
        exp = np.exp(decision - decision.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    return predict_fn


def render(model):
    st.title("Explication des prédictions (LIME)")
    st.markdown(
        "LIME identifie les mots les plus influents dans la prédiction du sentiment."
    )

    if model is None:
        st.error("Aucun modèle disponible. Exécutez le notebook 5.")
        st.stop()

    user_input = st.text_area(
        "Review à expliquer",
        height=150,
        value=(
            "The claim process was a nightmare. They denied my request without "
            "explanation and the customer service was rude."
        ),
    )
    num_features = st.slider("Nombre de mots à expliquer", 5, 20, 10)

    if not st.button("Expliquer avec LIME", type="primary"):
        return

    try:
        from lime.lime_text import LimeTextExplainer

        predict_fn = _build_predict_fn(model)
        explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)

        with st.spinner("Calcul LIME en cours..."):
            exp = explainer.explain_instance(
                user_input, predict_fn, num_features=num_features, num_samples=500
            )

        # Scores de confiance
        st.subheader("Scores de confiance")
        proba = predict_fn([user_input])[0]
        col1, col2, col3 = st.columns(3)
        for col, (cls, prob) in zip([col1, col2, col3], zip(CLASS_NAMES, proba)):
            with col:
                st.metric(cls.capitalize(), f"{prob:.1%}")

        pred_idx = np.argmax(proba)
        st.success(
            f"Prédiction : **{CLASS_NAMES[pred_idx].upper()}** "
            f"({proba[pred_idx]:.1%} de confiance)"
        )

        # Visualisation LIME
        st.subheader("Mots les plus influents")
        lime_html = exp.as_html()
        lime_html = lime_html.replace(
            "<body>",
            "<body style='background-color: white; color: black;'>",
        )
        st.components.v1.html(lime_html, height=400, scrolling=True)

        # Tableau des contributions
        st.subheader("Contributions par mot")
        lime_df = pd.DataFrame(exp.as_list(), columns=["Mot", "Contribution"])
        lime_df["Direction"] = lime_df["Contribution"].apply(
            lambda x: "pro-positif" if x > 0 else "pro-négatif"
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
