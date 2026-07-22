import base64
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SVM_DEPLOYMENT_MODEL_PATH


BACKGROUND_PATH = APP_DIR / "assets" / "banking_intent_background.png"


def set_background_image(image_path: Path) -> None:
    """Set the Streamlit page background image."""
    if not image_path.exists():
        return

    encoded_image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(2, 12, 27, 0.72),
                    rgba(2, 12, 27, 0.82)
                ),
                url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(5, 20, 42, 0.88);
            backdrop-filter: blur(12px);
        }}

        [data-testid="stMainBlockContainer"] {{
            background-color: rgba(7, 26, 52, 0.76);
            backdrop-filter: blur(14px);
            border-radius: 18px;
            padding: 2rem 2.5rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }}

        h1,
        h2,
        h3,
        p,
        label,
        .stMarkdown,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {{
            color: #f4f8ff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading calibrated Linear SVM model...")
def load_resources():
    """Load the deployment pipeline and label mapping."""
    model_path = Path(SVM_DEPLOYMENT_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            "Deployment model was not found: "
            f"{model_path}"
        )

    deployment = joblib.load(model_path)

    if "pipeline" not in deployment:
        raise KeyError(
            "The deployment bundle does not contain "
            "the 'pipeline' key."
        )

    if "label_mapping" not in deployment:
        raise KeyError(
            "The deployment bundle does not contain "
            "the 'label_mapping' key."
        )

    model = deployment["pipeline"]
    label_mapping = {
        int(label): intent
        for label, intent in deployment["label_mapping"].items()
    }

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not support predict_proba(). "
            "Train and save the pipeline with "
            "CalibratedClassifierCV."
        )

    return model, label_mapping


def predict_top_k(
    text: str,
    model: Any,
    label_mapping: dict[int, str],
    top_k: int = 3,
) -> dict[str, Any]:
    """Predict the most likely banking intents."""
    clean_text = text.strip()

    if not clean_text:
        raise ValueError(
            "Customer message must not be empty."
        )

    probabilities = model.predict_proba([clean_text])[0]
    classes = model.classes_
    top_k = min(top_k, len(classes))

    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    predictions = []

    for index in top_indices:
        label = int(classes[index])

        predictions.append(
            {
                "label": label,
                "intent": label_mapping.get(
                    label,
                    f"unknown_intent_{label}",
                ),
                "probability": float(probabilities[index]),
            }
        )

    best_prediction = predictions[0]

    return {
        "label": best_prediction["label"],
        "intent": best_prediction["intent"],
        "confidence": best_prediction["probability"],
        "top_predictions": predictions,
    }


def format_intent(intent: str) -> str:
    """Convert a raw intent label into a readable title."""
    return intent.replace("_", " ").title()


st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="centered",
)

set_background_image(BACKGROUND_PATH)

st.title("🏦 Banking Intent Classifier")

st.write(
    "Classify a customer support message into one of "
    "77 banking intents using a TF-IDF and calibrated "
    "Linear SVM model."
)

with st.sidebar:
    st.header("Model information")
    st.write("**Model:** TF-IDF + Calibrated Linear SVM")
    st.write("**Dataset:** Banking77")
    st.write("**Classes:** 77")
    st.write("**Calibration:** Sigmoid")
    st.write("**Original Linear SVM Test Macro F1:** 0.8879")

    st.divider()

    st.header("Best experimental model")
    st.write("**Model:** Fine-tuned DistilBERT")
    st.write("**Test Macro F1:** 0.9091")
    st.write(
        "**Hugging Face:** "
        "`foleg/banking-distilbert-intent-classifier`"
    )

    st.caption(
        "The calibrated Linear SVM is used for "
        "deployment because it requires significantly "
        "less memory than DistilBERT."
    )

    show_raw_labels = st.checkbox(
        "Show raw intent labels",
        value=False,
    )


example_messages = {
    "Custom message": "",
    "Card has not arrived": "My card has not arrived yet.",
    "Cash withdrawal fee": (
        "Why was I charged an extra fee when withdrawing cash?"
    ),
    "Transfer is pending": (
        "My bank transfer is still pending."
    ),
    "Identity verification": (
        "Why do I need to verify my identity?"
    ),
}

selected_example = st.selectbox(
    "Example",
    options=list(example_messages),
)

default_text = example_messages[selected_example]

customer_message = st.text_area(
    "Customer message",
    value=default_text,
    placeholder="Enter a banking support request...",
    height=150,
)

classify_button = st.button(
    "Classify intent",
    type="primary",
    use_container_width=True,
)

if classify_button:
    if not customer_message.strip():
        st.warning("Enter a customer message first.")
    else:
        try:
            model, label_mapping = load_resources()

            with st.spinner("Classifying message..."):
                result = predict_top_k(
                    text=customer_message,
                    model=model,
                    label_mapping=label_mapping,
                    top_k=3,
                )

            predicted_intent = result["intent"]

            display_intent = (
                predicted_intent
                if show_raw_labels
                else format_intent(predicted_intent)
            )

            st.success(
                f"Predicted intent: **{display_intent}**"
            )

            st.metric(
                "Predicted probability",
                f"{result['confidence']:.2%}",
            )

            st.subheader("Top predictions")

            for index, prediction in enumerate(
                result["top_predictions"],
                start=1,
            ):
                intent = prediction["intent"]
                probability = prediction["probability"]

                display_label = (
                    intent
                    if show_raw_labels
                    else format_intent(intent)
                )

                st.write(
                    f"**{index}. {display_label}** "
                    f"— {probability:.2%}"
                )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

            if result["confidence"] < 0.60:
                st.warning(
                    "The model confidence is relatively low. "
                    "This request may require manual review."
                )

        except Exception as exc:
            st.error(
                "Prediction failed. "
                f"Technical details: {exc}"
            )
