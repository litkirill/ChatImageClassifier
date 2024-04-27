from .ocr_model import extract_text_from_image
from .classify_text_model import classify_text
from .types import LabelType


def predict_model(uploaded_file: str) -> LabelType:
    """Predicts whether an image contains text that classifies it as a chat
    screenshot."""
    text = extract_text_from_image(uploaded_file)
    predicted = classify_text(text)

    return LabelType.CHAT if "<chat>" in predicted else LabelType.NOT_CHAT
