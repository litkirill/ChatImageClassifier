from typing import Optional
import base64
import json
import os
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

CATALOG_ID = os.getenv('CATALOG_ID')
YANDEX_OCR_API_KEY = os.getenv('YANDEX_GPT_API_KEY')

if not CATALOG_ID or not YANDEX_OCR_API_KEY:
    logger.error("Critical environment variables are missing.")
    exit(1)

url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

logger.add("debug.log", format="{time} {level} {message}", level="DEBUG")


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encodes an image to a base64 string."""
    try:
        image_file = open(image_path, "rb").read()
        encoded_image = base64.b64encode(image_file).decode('utf-8')
        logger.debug("Image encoded successfully")
        return encoded_image
    except FileNotFoundError:
        logger.error(f"File not found - {image_path}")
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
        logger.debug("API call was successful")
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error occurred: {e}")
        return None
    except requests.exceptions.Timeout as e:
        logger.error("The request timed out: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return None


def extract_text_from_ocr_response(ocr_result: dict) -> Optional[str]:
    """Extracts and returns the full text from an OCR API response."""
    try:
        full_text = ocr_result["result"]["textAnnotation"]["fullText"]
        logger.debug("Text extracted successfully")
        return full_text
    except Exception as e:
        logger.error(f"An error occurred while extracting text: {e}")
        return None
