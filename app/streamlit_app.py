import base64
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DISTILBERT_MODEL_ID


BACKGROUND_PATH = (
    APP_DIR
    / "assets"
    / "banking_intent_background.png"
)

MAX_LENGTH = 128


def set_background_image(
    image_path: Path,
) -> None:
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
                url(
                    "data:image/png;base64,{encoded_image}"
                );
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(
                5,
                20,
                42,
                0.88
            );
            backdrop-filter: blur(12px);
        }}

        [data-testid="stMainBlockContainer"] {{
            background-color: rgba(
                7,
                26,
                52,
                0.76
            );
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


st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="centered",
)

set_background_image(BACKGROUND_PATH)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        DISTILBERT_MODEL_ID
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_ID
    )

    device = get_device()

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_top_k(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    top_k: int = 3,
) -> dict[str, Any]:
    clean_text = text.strip()

    if not clean_text:
        raise ValueError("Customer message must not be empty.")

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    top_k = min(top_k, model.config.num_labels)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
    )

    predictions = []

    for probability, label_id in zip(
        top_probabilities,
        top_indices,
    ):
        numeric_label = int(label_id.item())

        predictions.append(
            {
                "label": numeric_label,
                "intent": model.config.id2label[numeric_label],
                "probability": float(probability.item()),
            }
        )

    return {
        "intent": predictions[0]["intent"],
        "confidence": predictions[0]["probability"],
        "top_predictions": predictions,
    }


def format_intent(intent: str) -> str:
    return intent.replace("_", " ").title()

st.title("🏦 Banking Intent Classifier")

st.write(
    "Classify a customer support message into one of "
    "77 banking intents using a fine-tuned DistilBERT model."
)

with st.sidebar:
    st.header("Model information")
    st.write("**Model:** DistilBERT")
    st.write("**Dataset:** Banking77")
    st.write("**Classes:** 77")
    st.write("**Test Macro F1:** 0.9091")
    st.write("**Source:** Hugging Face Hub")

    show_raw_labels = st.checkbox(
        "Show raw intent labels",
        value=False,
    )

example_messages = {
    "Custom message": "",
    "Card has not arrived": (
        "My card has not arrived yet."
    ),
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
            tokenizer, model, device = load_model()

            with st.spinner("Classifying message..."):
                result = predict_top_k(
                    text=customer_message,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
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
                "Confidence",
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