"""
Page 6 — Question Answering extractif.
L'utilisateur pose une question, le système cherche les reviews pertinentes
puis extrait la réponse directement depuis le texte des reviews.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import FAISS_INDEX_DIR, SENTENCE_EMBEDDINGS_DIR


@st.cache_resource
def _load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def _load_faiss_index():
    import faiss
    if not FAISS_INDEX_DIR.exists():
        return None
    return faiss.read_index(str(FAISS_INDEX_DIR))


@st.cache_data
def _load_embeddings():
    if not SENTENCE_EMBEDDINGS_DIR.exists():
        return None
    return np.load(str(SENTENCE_EMBEDDINGS_DIR))


@st.cache_resource
def _load_qa_pipeline():
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model


def _run_qa(question: str, context: str) -> dict:
    """Extraction de réponse avec RoBERTa-SQuAD2 sans pipeline."""
    import torch
    tokenizer, model = _load_qa_pipeline()
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    tokens = inputs["input_ids"][0][start:end]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    # Score approximatif
    score = float(
        torch.softmax(outputs.start_logits, dim=-1)[0][start]
        * torch.softmax(outputs.end_logits, dim=-1)[0][end - 1]
    )
    return {"answer": answer, "score": score}


def _retrieve_contexts(query: str, df: pd.DataFrame, top_k: int) -> list[dict]:
    """Récupère les top_k reviews les plus proches via FAISS."""
    sbert = _load_sbert()
    index = _load_faiss_index()
    embeddings = _load_embeddings()

    if index is None or embeddings is None:
        import faiss
        text_col = "review_clean" if "review_clean" in df.columns else "review"
        corpus = df[text_col].fillna("").tolist()
        embeddings = sbert.encode(corpus, show_progress_bar=False, batch_size=64)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))

    query_vec = sbert.encode([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    distances, indices = index.search(query_vec.astype(np.float32), top_k)

    results = []
    display_col = "review" if "review" in df.columns else "review_clean"
    for idx, score in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        results.append({
            "score": float(score),
            "review": str(row.get(display_col, "")),
            "stars": row.get("stars", None),
            "insurer": row.get("insurer", None),
            "sentiment": row.get("sentiment", None),
        })
    return results


def render(df: pd.DataFrame):
    st.title("Question Answering — Extraction de réponses")
    st.markdown(
        "Posez une question sur les avis d'assurance. "
        "Le système récupère les reviews pertinentes via **FAISS + Sentence-BERT**, "
        "puis extrait la réponse directement depuis le texte avec un modèle de **QA extractif** "
        "(RoBERTa fine-tuné sur SQuAD 2)."
    )

    if df is None:
        st.error("Dataset non chargé. Exécutez les notebooks 1 et 2.")
        st.stop()

    top_k = st.slider("Nombre de reviews à analyser", 3, 15, 5, key="qa_topk")

    query = st.text_input(
        "Votre question",
        placeholder="Ex: Quel est le principal problème avec les remboursements ?",
        key="qa_query",
    )

    if st.button("Trouver la réponse", type="primary", key="qa_btn") and query.strip():
        with st.spinner("Recherche des reviews pertinentes..."):
            contexts = _retrieve_contexts(query, df, top_k)

        if not contexts:
            st.warning("Aucun résultat trouvé.")
            st.stop()

        # Extraire une réponse de chaque review
        st.subheader("Réponses extraites")
        combined_context = " ".join(c["review"][:500] for c in contexts[:5])

        # Réponse globale sur le contexte combiné
        with st.spinner("Extraction de la réponse..."):
            try:
                global_answer = _run_qa(query, combined_context[:2000])
            except Exception:
                global_answer = None

        if global_answer and global_answer.get("answer"):
            st.success(
                f"**Réponse :** {global_answer['answer']}  \n"
                f"**Confiance :** {global_answer['score']:.1%}"
            )
        else:
            st.warning("Le modèle n'a pas pu extraire de réponse claire.")

        # Détails par review
        st.subheader("Réponses par review source")
        for i, ctx in enumerate(contexts):
            try:
                ans = _run_qa(query, ctx["review"][:1500])
            except Exception:
                ans = {"answer": "—", "score": 0.0}

            stars_str = f"{int(ctx['stars'])} étoiles" if ctx["stars"] else ""
            insurer_str = str(ctx["insurer"] or "")
            header = f"#{i+1} — Confiance : {ans['score']:.1%}"
            if stars_str:
                header += f" | {stars_str}"
            if insurer_str:
                header += f" | {insurer_str}"

            with st.expander(header, expanded=(i == 0)):
                if ans["answer"] and ans["answer"] != "—":
                    st.markdown(f"**Réponse extraite :** {ans['answer']}")
                else:
                    st.info("Pas de réponse extraite pour cette review.")
                st.markdown("**Review :**")
                st.write(ctx["review"][:500])
