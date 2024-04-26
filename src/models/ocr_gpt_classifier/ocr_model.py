import easyocr
from PIL import Image

reader = easyocr.Reader(['ru', 'en'], gpu=False)


def extract_text(image: Image.Image) -> str:
    """
    Extracts and concatenates text from an image using OCR.
    """

    results = reader.readtext(image, detail=0, paragraph=True)

    return " ".join([result for result in results])