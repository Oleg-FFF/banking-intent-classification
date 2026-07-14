from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE


def build_logistic_pipeline(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = None,
    c: float = 1.0,
) -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression classification pipeline.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_svm_pipeline(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = None,
    c: float = 1.0,
) -> Pipeline:
    """
    Build a TF-IDF + Linear SVM classification pipeline.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=c,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )