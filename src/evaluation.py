import json
from pathlib import Path
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


def save_experiment_result(
    result: dict[str, Any],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Add or update one experiment result in a shared CSV file.

    A result is uniquely identified by:
    model + variant + split.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_result = result.copy()

    parameters = prepared_result.get("parameters")

    if isinstance(parameters, dict):
        prepared_result["parameters"] = json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
        )

    new_row = pd.DataFrame([prepared_result])

    if output_path.exists():
        results_df = pd.read_csv(output_path)
        results_df = pd.concat(
            [results_df, new_row],
            ignore_index=True,
        )
    else:
        results_df = new_row

    key_columns = [
        column
        for column in ["model", "variant", "split"]
        if column in results_df.columns
    ]

    if key_columns:
        results_df = results_df.drop_duplicates(
            subset=key_columns,
            keep="last",
        )

    if "macro_f1" in results_df.columns:
        results_df = results_df.sort_values(
            "macro_f1",
            ascending=False,
        )

    results_df = results_df.reset_index(drop=True)

    results_df.to_csv(
        output_path,
        index=False,
    )

    return results_df