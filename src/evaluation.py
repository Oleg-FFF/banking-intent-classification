from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    model_name: str,
    y_true,
    y_pred,
    train_time: float | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate the main multiclass classification metrics.
    """
    return {
        "model": model_name,
        "parameters": parameters,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "train_time_sec": train_time,
    }


def create_results_table(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert experiment results into a sorted DataFrame.
    """
    return (
        pd.DataFrame(results)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )


def print_classification_report(
    y_true,
    y_pred,
    target_names: list[str] | None = None,
) -> None:
    """
    Print a readable sklearn classification report.
    """
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )
    )