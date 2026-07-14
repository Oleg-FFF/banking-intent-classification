from pathlib import Path


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MODEL_RESULTS_PATH = METRICS_DIR / "model_results.csv"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = PROCESSED_DATA_DIR / "train_preprocessed.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation_preprocessed.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test_preprocessed.csv"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_pipeline.joblib"
SVM_MODEL_PATH = MODELS_DIR / "svm_pipeline.joblib"

TEXT_COLUMN = "text_clean"
TARGET_COLUMN = "label"
TARGET_NAME_COLUMN = "label_text"