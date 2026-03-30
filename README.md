# Analyse d'avis clients - Assurances

Analyse NLP d'avis clients d'assureurs : topic modeling (LDA), word embeddings (Word2Vec, sentence-transformers),
classification de sentiment (TF-IDF + LR/SVM, réseau de neurones), et recherche sémantique (FAISS).

## Installation

```bash
./run.sh
```

> Le script vérifie Python 3.12, crée le venv, installe les dépendances et télécharge le modèle spaCy.
> TensorFlow s'installe automatiquement selon la plateforme (Mac Apple Silicon ou Linux/Windows).

Une fois les dépendances installées, il n'est pas obligatoire d'attendre la fin de l'exécution des notebooks pour lancer l'application,
mais cela est fortement conseillé car le modèle utilisé pour la prédiction par défaut n'est pas super efficace.

> Pour coder avec un IDE, il faut configurer l'environnement d'exécution python qui se trouve dans `./venv/` selon l'IDE utilisé.



## Utilisation

**Lancer activer l'env :**

```bash
source venv/bin/activate
```
**Lancer l'application :**

```bash
streamlit run app/streamlit_app.py
```

**Onglets disponibles :**

- `Prediction` — saisir un avis, choisir un modèle (LR, SVM, NN), obtenir la note prédite + explication LIME
- `Résumé par assureur` — statistiques et distribution des notes par compagnie
- `Explication LIME` — analyse de l'impact des mots sur la prédiction pour un avis du dataset
- `Recherche IR` — trouver les avis les plus similaires à une requête (FAISS)

**Notebooks (ordre d'exécution) :**

```
1_exploration.ipynb             → analyse exploratoire
2_cleaning_preprocessing.ipynb  → nettoyage et prétraitement
3_topic_modeling.ipynb          → LDA
4_embeddings.ipynb              → Word2Vec + sentence-transformers
5_supervised_learning.ipynb     → classification + évaluation
```
