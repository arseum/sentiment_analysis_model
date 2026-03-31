"""
Application Streamlit - Analyse d'avis d'assurance
Projet 2 NLP 2026 - Arsène Maitre & Gabriel Thibout
"""

import sys
from pathlib import Path

import streamlit as st

from utils.loaders import load_data, load_model
from part import page_prediction, page_insurer, page_lime, page_ir, page_rag, page_qa

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Prediction humeur message",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.sidebar.title("Prediction d'humeur")
st.sidebar.markdown("NLP Projet 2 — 2026")
st.sidebar.markdown("Arsène Maitre & Gabriel Thibout")
st.sidebar.divider()

part = st.sidebar.radio(
    "Navigation",
    [
        "Prédiction",
        "Résumé par assureur",
        "Explication LIME",
        "Recherche IR",
        "RAG",
        "Question Answering",
    ],
)

df = load_data()
model, model_name = load_model()

if part == "Prédiction":
    page_prediction.render(model, model_name)

elif part == "Résumé par assureur":
    page_insurer.render(df)

elif part == "Explication LIME":
    page_lime.render(model)

elif part == "Recherche IR":
    page_ir.render(df)

elif part == "RAG":
    page_rag.render(df)

elif part == "Question Answering":
    page_qa.render(df)
