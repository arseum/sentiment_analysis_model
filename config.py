from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
DATA_PATH = DATA_PROCESSED / 'data.csv'
DATA_CLEAN = DATA_PROCESSED / 'reviews_clean.csv'

MODELS_DIR = BASE_DIR / 'models'

LDA_DIR = MODELS_DIR / 'lda'
LDA_MODELS_DIR = LDA_DIR / 'lda_model'
DICTIONARIES_DIR = LDA_DIR / 'lda_dictionary'
LDA_VIZ_PATH = LDA_DIR / 'lda_visualization.html'

W2V_DIR = MODELS_DIR / 'word2vec'
W2V_MODEL_PATH = W2V_DIR / 'word2vec.model'

SBERT_DIR = MODELS_DIR / 'sbert'
FAISS_INDEX_DIR = SBERT_DIR / 'faiss_index.bin'
SENTENCE_EMBEDDINGS_DIR = SBERT_DIR / 'sentence_embeddings.npy'

SUPERVISED_DIR = MODELS_DIR / 'supervised'
TFIDF_LR_PATH = SUPERVISED_DIR / 'tfidf_lr.pkl'
TFIDF_SVM_PATH = SUPERVISED_DIR / 'tfidf_svm.pkl'
NN_EMBEDDING_DIR = SUPERVISED_DIR / 'nn_embedding'
NN_EMBEDDING_MODEL_PATH = NN_EMBEDDING_DIR / 'model_nn.keras'
NN_W2V_DIR = SUPERVISED_DIR / 'nn_word2vec'
NN_W2V_MODEL_PATH = NN_W2V_DIR / 'model_w2v.keras'
BERT_DIR = SUPERVISED_DIR / 'bert_sentiment'

LOG_DIR = MODELS_DIR / 'tensorboard_logs'
TENSORBOARD_DIR = LOG_DIR / 'embeddings'
VECTORS_PATH = TENSORBOARD_DIR / 'vectors.tsv'
METADATA_PATH = TENSORBOARD_DIR / 'metadata.tsv'

for d in [LDA_DIR, W2V_DIR, SBERT_DIR, SUPERVISED_DIR,
          NN_EMBEDDING_DIR, NN_W2V_DIR, BERT_DIR, TENSORBOARD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TEXT_COL = 'avis'
RATING_COL = 'note'
INSURER_COL = 'assureur'
DATE_COL = 'date_publication'
LENGTH_COL = 'avis_taille'
CLEAN_COL = 'avis_clean'
SENTIMENT_COL = 'sentiment'

TOPIC_LABELS = {
    0: 'Sinistres Auto',
    1: 'Problèmes & Litiges',
    2: 'Tarifs & Contrats',
    3: 'Assurance Santé',
    4: 'Prise en Charge',
    5: 'Satisfaction Client',
}

# Labels zero-shot alignés sur les topics LDA
ZS_CATEGORY_LABELS = [
    'sinistres et accidents',
    'problèmes et litiges',
    'tarifs et contrats',
    'assurance santé et mutuelle',
    'prise en charge et assistance',
    'satisfaction et service client',
]
