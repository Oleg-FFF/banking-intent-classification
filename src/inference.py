from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocessing import minimal_preprocess


class BankingIntentPredictor:
    """
    Load a trained sklearn pipeline and perform intent predictions.
    """

    def __init__(
        self,
        model_path: str | Path,
        label_mapping: dict[int, str],
    ) -> None:
        self.model_path = Path(model_path)
        self.label_mapping = label_mapping
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        return joblib.load(self.model_path)

    def predict(self, text: str) -> dict[str, Any]:
        cleaned_text = minimal_preprocess(text)

        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        predicted_label = int(
            self.model.predict([cleaned_text])[0]
        )

        return {
            "label": predicted_label,
            "intent": self.label_mapping[predicted_label],
        }

    def predict_top_k(
        self,
        text: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        cleaned_text = minimal_preprocess(text)

        if not cleaned_text:
            raise ValueError("Text must not be empty.")

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "The current model does not support predict_proba()."
            )

        probabilities = self.model.predict_proba(
            [cleaned_text]
        )[0]

        classes = self.model.classes_
        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        predictions = [
            {
                "label": int(classes[index]),
                "intent": self.label_mapping[int(classes[index])],
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ]

        return {
            "intent": predictions[0]["intent"],
            "confidence": predictions[0]["probability"],
            "top_predictions": predictions,
        }