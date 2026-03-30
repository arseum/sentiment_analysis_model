#!/bin/bash
set -e

# Vérifier si Python 3.12 est installé
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 n'est pas installé."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Installe-le avec : brew install python@3.12"
    else
        echo "Installe-le depuis : https://www.python.org/downloads/"
    fi
    exit 1
fi

echo "Python 3.12 trouvé : $(python3.12 --version)"

# Créer le venv si inexistant
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3.12 -m venv venv
fi

# Activer le venv
echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "Installation des dépendances..."
pip install --upgrade pip -q
pip install -r requirements.txt

# Télécharger le modèle spaCy
echo "Téléchargement du modèle spaCy..."
python -m spacy download fr_core_news_sm

echo ""
echo "Setup terminé."

# Exécuter les notebooks dans l'ordre
echo "Lancement des notebooks..."
for nb in notebooks/1_exploration.ipynb \
           notebooks/2_cleaning_preprocessing.ipynb \
           notebooks/3_topic_modeling.ipynb \
           notebooks/4_embeddings.ipynb \
           notebooks/5_supervised_learning.ipynb; do
    echo ""
    echo ">>> Exécution de $nb ..."
    # Nettoyer la métadonnée JetBrains invalide
    python -c "
import json
with open('$nb') as f: nb = json.load(f)
nb.get('metadata', {}).pop('jetTransient', None)
with open('$nb', 'w') as f: json.dump(nb, f, indent=1)
"
    PYTHONPATH="$(pwd)" jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 \
        --ExecutePreprocessor.kernel_name=python3 "$nb"
    echo "<<< $nb terminé."
done

echo ""
echo "Tous les notebooks ont été exécutés."
