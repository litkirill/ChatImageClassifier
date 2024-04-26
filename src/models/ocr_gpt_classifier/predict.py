from PIL import Image
from .ocr_model import extract_text
from .classify_text_model import classify_text
from .types import LabelType


def predict_model(image: Image.Image) -> LabelType:
    """
     Predicts whether an image contains text that classifies it as a chat screenshot.
    """

    text = extract_text(image)
    predicted = classify_text(text)

    return LabelType.CHAT if "<chat>" in predicted else LabelType.NOT_CHAT
