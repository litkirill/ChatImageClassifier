import base64
import json
import requests
from typing import IO, Optional
from src.config import logger, \
    CATALOG_ID, YANDEX_OCR_API_KEY, YANDEX_OCR_URL


def encode_image_to_base64(uploaded_file: IO[bytes]) -> Optional[str]:
    """Encodes an image to a base64 string."""
    try:
        image_bytes: bytes = uploaded_file.read()
        encoded_image: str = base64.b64encode(image_bytes).decode('utf-8')
        logger.debug("Image encoded successfully")
        return encoded_image
    except FileNotFoundError:
        logger.error(f"Image file not found")
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
        response = requests.post(YANDEX_OCR_URL, headers=headers,
                                 data=request_body)
        response.raise_for_status()  # Raises an HTTPError for bad requests
        logger.info("OCR API call was successful")
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send request: {e}",
                     exc_info=True)  # Log traceback
        raise ConnectionError(f"Failed to send request OCR API: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the OCR API call: {e}")
        return None


def extract_text_from_ocr_response(ocr_result: dict) -> Optional[str]:
    """Extracts and returns the full text from an OCR API response."""
    try:
        full_text = ocr_result["result"]["textAnnotation"]["fullText"]
        logger.info("Text extracted successfully")
        return full_text
    except Exception as e:
        logger.error(f"An error occurred while extracting text: {e}")
        return None


def extract_text_from_image(uploaded_file: IO[bytes]) -> Optional[str]:
    """Extracts text from an image using the Yandex OCR API."""
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

    logger.info(
        f"Text classified successfully: {extracted_text.replace("\n", "\\n")}"
    )

    return extracted_text
