import base64
import json
import os
import requests
from typing import IO, Optional
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CATALOG_ID = os.getenv('CATALOG_ID')
YANDEX_OCR_API_KEY = os.getenv('YANDEX_OCR_API_KEY')

if not CATALOG_ID or not YANDEX_OCR_API_KEY:
    logger.error("Critical environment variables are missing.")
    raise EnvironmentError("Critical environment variables are missing. Please check CATALOG_ID and YANDEX_OCR_API_KEY.")

url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

logger.add(
    "logs/logs_from_app.log", format="{time} {level} {message}",
    level="DEBUG", rotation="10 MB", compression="zip"
)


def encode_image_to_base64(uploaded_file: IO[bytes]) -> Optional[str]:
    """Encodes an image to a base64 string."""
    try:
        image_bytes: bytes = uploaded_file.read()
        encoded_image: str = base64.b64encode(image_bytes).decode('utf-8')
        logger.debug("Image encoded successfully")
        return encoded_image
    except FileNotFoundError:
        logger.error(f"File not found")
        return None


def create_request_body(image_base64: Optional[str]) -> Optional[str]:
    """Creates a JSON request body for sending to the Yandex OCR API."""
    if image_base64 is None:
        logger.warning("No image data to encode.")
        return None

    body = {
        "languageCodes": ["ru", "en"],
        "model": "page",
        "content": image_base64
    }
    request_body = json.dumps(body)
    logger.debug("Request body created successfully")
    return request_body


def call_ocr_api(request_body: str) -> Optional[dict]:
    """Calls the OCR API and returns the result."""
    if request_body is None:
        logger.warning("Request body is empty.")
        return None
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_OCR_API_KEY}",
            "x-folder-id": CATALOG_ID,
            "x-data-logging-enabled": "true"
        }
        response = requests.post(url, headers=headers, data=request_body)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        logger.debug("OCR API call was successful")
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to send request: {e}")


def extract_text_from_ocr_response(ocr_result: dict) -> Optional[str]:
    """Extracts and returns the full text from an OCR API response."""
    try:
        full_text = ocr_result["result"]["textAnnotation"]["fullText"]
        logger.debug("Text extracted successfully")
        return full_text
    except Exception as e:
        logger.error(f"An error occurred while extracting text: {e}")
        return None


def extract_text_from_image(uploaded_file: IO[bytes]) -> Optional[str]:
    """Extracts text from an image using the Yandex OCR API."""
    print(type(uploaded_file))

    image_base64 = encode_image_to_base64(uploaded_file)
    if not image_base64:
        logger.error("Failed to encode the image.")
        return None

    request_body = create_request_body(image_base64)
    if not request_body:
        logger.error("Failed to create the request body.")
        return None

    ocr_result = call_ocr_api(request_body)
    if not ocr_result:
        logger.error("OCR API call failed.")
        return None

    extracted_text = extract_text_from_ocr_response(ocr_result)
    if not extracted_text:
        logger.error("Failed to extract text from the OCR response.")
        return None

    return extracted_text
