"""
Page 5 — RAG (Retrieval-Augmented Generation).
Récupère les reviews les plus pertinentes via FAISS + Sentence-BERT,
puis génère une réponse avec Flan-T5 (pipeline de traduction FR→EN→FR si nécessaire).
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
def _load_translator_fr_en():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")


@st.cache_resource
def _load_translator_en_fr():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")


@st.cache_resource
def _load_generator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


def _translate_fr_en(text: str) -> str:
    translator = _load_translator_fr_en()
    result = translator(text[:512], max_length=512)
    return result[0]["translation_text"]


def _translate_en_fr(text: str) -> str:
    translator = _load_translator_en_fr()
    result = translator(text[:512], max_length=512)
    return result[0]["translation_text"]


def _retrieve(query: str, df: pd.DataFrame, top_k: int):
    """Retourne les top_k reviews les plus proches via FAISS."""
    sbert = _load_sbert()
    index = _load_faiss_index()
    embeddings = _load_embeddings()

    if index is None or embeddings is None:
        st.info("Index FAISS non trouvé — construction à la volée...")
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
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        results.append({
            "rank": rank + 1,
            "score": float(score),
            "review": str(row.get(display_col, "")),
            "stars": row.get("stars", None),
            "insurer": row.get("insurer", None),
            "sentiment": row.get("sentiment", None),
        })
    return results


def _generate_answer(query: str, contexts: list[dict]) -> str:
    """
    Traduit le contexte FR→EN, génère une réponse avec Flan-T5, traduit EN→FR.
    """
    import torch

    # Traduire les reviews et la question en anglais
    translated_reviews = []
    for c in contexts[:5]:
        try:
            translated_reviews.append(_translate_fr_en(c["review"][:400]))
        except Exception:
            translated_reviews.append(c["review"][:400])

    try:
        query_en = _translate_fr_en(query)
    except Exception:
        query_en = query

    context_text = "\n".join(
        f"- (rating {c.get('stars', '?')}/5) {rev}"
        for c, rev in zip(contexts[:5], translated_reviews)
    )

    prompt = (
        f"Based on these insurance customer reviews, answer the question with a complete sentence.\n\n"
        f"Reviews:\n{context_text}\n\n"
        f"Question: {query_en}\n\n"
        f"Answer:"
    )

    tokenizer, model = _load_generator()
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
        )
    answer_en = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Traduire la réponse EN→FR
    try:
        answer_fr = _translate_en_fr(answer_en)
    except Exception:
        answer_fr = answer_en

    return answer_fr


def _synthesize(contexts: list[dict]) -> dict:
    """Synthèse extractive structurée (métriques + citations)."""
    positives = [c for c in contexts if str(c.get("sentiment", "")).lower() == "positif"]
    negatives = [c for c in contexts if str(c.get("sentiment", "")).lower() == "négatif"]
    neutrals  = [c for c in contexts if str(c.get("sentiment", "")).lower() == "neutre"]

    stars = [c["stars"] for c in contexts if c.get("stars") is not None]
    avg_stars = sum(stars) / len(stars) if stars else None

    def best_quote(group):
        if not group:
            return None
        sorted_group = sorted(group, key=lambda c: len(c["review"]))
        for c in sorted_group:
            text = c["review"].strip()
            if 30 < len(text) < 400:
                return text
        return sorted_group[0]["review"][:300]

    return {
        "avg_stars": avg_stars,
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "n_neu": len(neutrals),
        "quote_pos": best_quote(positives),
        "quote_neg": best_quote(negatives),
        "quote_neu": best_quote(neutrals),
    }


def render(df: pd.DataFrame):
    st.title("RAG — Génération augmentée par récupération")
    st.markdown(
        "Posez une question sur les avis d'assurance. "
        "Le système récupère les reviews les plus pertinentes via **FAISS + Sentence-BERT** *(Retrieval)*, "
        "traduit le contexte FR→EN, génère une réponse avec **Flan-T5** *(Augmented Generation)*, "
        "puis retraduit EN→FR via **Helsinki-NLP**."
    )

    if df is None:
        st.error("Dataset non chargé. Exécutez les notebooks 1 et 2.")
        st.stop()

    top_k = st.slider("Nombre de reviews à récupérer", 3, 15, 5)

    query = st.text_input(
        "Votre question",
        placeholder="Ex: Que pensent les clients du service client en général ?",
    )

    if st.button("Générer une réponse", type="primary") and query.strip():
        with st.spinner("Récupération des reviews pertinentes (FAISS + SBERT)..."):
            results = _retrieve(query, df, top_k)

        if not results:
            st.warning("Aucun résultat trouvé.")
            st.stop()

        # --- Réponse générée (RAG) ---
        with st.spinner("Traduction FR→EN + génération Flan-T5 + traduction EN→FR..."):
            try:
                answer = _generate_answer(query, results)
                st.subheader("Réponse générée (RAG)")
                st.success(answer)
            except Exception as e:
                st.warning(f"Génération indisponible : {e}")

        # --- Synthèse extractive ---
        st.subheader("Synthèse des avis récupérés")
        synth = _synthesize(results)
        total = synth["n_pos"] + synth["n_neg"] + synth["n_neu"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Reviews analysées", total)
        with c2:
            stars_label = f"{synth['avg_stars']:.1f} / 5" if synth["avg_stars"] else "—"
            st.metric("Note moyenne", stars_label)
        with c3:
            pct_pos = synth["n_pos"] / total * 100 if total else 0
            st.metric("Avis positifs", f"{synth['n_pos']} ({pct_pos:.0f}%)")
        with c4:
            pct_neg = synth["n_neg"] / total * 100 if total else 0
            st.metric("Avis négatifs", f"{synth['n_neg']} ({pct_neg:.0f}%)")

        if synth["quote_pos"]:
            st.markdown("**Ce que les clients apprécient :**")
            st.success(f'"{synth["quote_pos"]}"')
        if synth["quote_neg"]:
            st.markdown("**Ce que les clients critiquent :**")
            st.error(f'"{synth["quote_neg"]}"')
        if synth["quote_neu"]:
            st.markdown("**Avis nuancé :**")
            st.info(f'"{synth["quote_neu"]}"')

        # --- Sources ---
        st.subheader("Sources (reviews récupérées)")
        for c in results:
            stars_str = f"{int(c['stars'])} étoiles" if c["stars"] else ""
            insurer_str = str(c["insurer"] or "")
            sentiment_str = str(c["sentiment"] or "").capitalize()
            header = f"#{c['rank']} — Score : {c['score']:.3f}"
            if stars_str:
                header += f" | {stars_str}"
            if insurer_str:
                header += f" | {insurer_str}"
            if sentiment_str:
                header += f" | {sentiment_str}"
            with st.expander(header, expanded=(c["rank"] == 1)):
                st.write(c["review"][:500])
