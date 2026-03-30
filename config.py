from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
DATA_PATH = DATA_PROCESSED / 'data.csv'

DATA_CLEAN = DATA_PROCESSED / 'reviews_clean.csv'

MODELS_DIR = BASE_DIR / 'models'
LDA_MODELS_DIR = MODELS_DIR / 'lda_model'
DICTIONARIES_DIR = BASE_DIR / 'lda_dictionary'
LOG_DIR = BASE_DIR / 'tensorboard_logs'
FAISS_INDEX_DIR = MODELS_DIR / 'faiss_index.bin'
SENTENCE_EMBEDDINGS_DIR = MODELS_DIR / 'sentence_embeddings.npy'
MODELS_DIR.mkdir(exist_ok=True)

TENSORBOARD_DIR = MODELS_DIR / 'tensorboard_logs' / 'embeddings'
VECTORS_PATH = TENSORBOARD_DIR / 'vectors.tsv'
METADATA_PATH = TENSORBOARD_DIR / 'metadata.tsv'

TOPIC_LABELS = {
    0: 'Sinistres Auto',
    1: 'Problèmes & Litiges',
    2: 'Tarifs & Contrats',
    3: 'Assurance Santé',
    4: 'Prise en Charge',
    5: 'Satisfaction Client',
}

TEXT_COL = 'avis'
RATING_COL = 'note'
INSURER_COL = 'assureur'
DATE_COL = 'date_publication'
LENGTH_COL = 'avis_taille'
CLEAN_COL = 'avis_clean'
