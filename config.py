from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
DATA_CLEAN = DATA_PROCESSED / 'reviews_clean.csv'

MODELS_DIR.mkdir(exist_ok=True)

TEXT_COL = 'avis'
RATING_COL = 'note'
INSURER_COL = 'assureur'
DATE_COL = 'date_publication'
LENGTH_COL = 'avis_taille'
CLEAN_COL = 'avis_clean'
