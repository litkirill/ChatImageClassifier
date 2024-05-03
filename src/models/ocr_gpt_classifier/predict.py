"""Module for predicting text characteristics from images using OCR and text classification.

This module integrates OCR (Optical Character Recognition) and text classification to determine
if an image contains text that classifies it as a chat screenshot. It uses specialized
modules for OCR processing and text classification.
"""

from typing import Optional
from src.config import logger
from .ocr_model import extract_text_from_image
from .classify_text_model import classify_text
from .types import LabelType


def predict_model(uploaded_file) -> Optional[LabelType]:
    """Predicts whether an image contains text that classifies it as a chat
    screenshot or not."""
    text = extract_text_from_image(uploaded_file)
    if text is None or text.strip() == "":
        logger.info("No text found on the image.")
        return LabelType.NOT_CHAT
    predicted = classify_text(text)
    return LabelType.CHAT if "<chat>" in predicted else LabelType.NOT_CHAT
