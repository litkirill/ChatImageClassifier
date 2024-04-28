from loguru import logger
from typing import Optional
from .ocr_model import extract_text_from_image
from .classify_text_model import classify_text
from .types import LabelType

logger.add(
    "logs/logs_from_app.log", format="{time} {level} {message}",
    level="DEBUG", rotation="10 MB", compression="zip"
)


def predict_model(uploaded_file) -> Optional[LabelType]:
    """Predicts whether an image contains text that classifies it as a chat
    screenshot or not."""
    try:
        text = extract_text_from_image(uploaded_file)
        if text is None or text.strip() == "":
            logger.info("No text found on the image.")
            return LabelType.NOT_CHAT
        predicted = classify_text(text)
        return LabelType.CHAT if "<chat>" in predicted else LabelType.NOT_CHAT
    except Exception as e:
        logger.error(f"Failed to process the image: {e}")
        return None
