import re


def minimal_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Нормалізація апострофів
    text = text.replace("’", "'").replace("`", "'")

    # Прибираємо переноси рядків і табуляції
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Нормалізуємо пробіли
    text = re.sub(r"\s+", " ", text)

    return text.strip()