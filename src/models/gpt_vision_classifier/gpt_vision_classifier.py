"""Module for classifying images into two class:
<chat> and <non-chat> using OpenAI's GPT model."""

import os
import base64
from typing import IO, Optional, Tuple
import yaml
from PIL import Image
import requests

from src.config import logger, \
    OPENAI_API_KEY, GPT_URL, GPT_VERSION, DEFAULT_IMAGE_SIZE, MAX_TOKENS


def resize_image(
        file: IO[bytes],
        size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Optional[Image.Image]:
    """Resizes an image to the specified size."""
    try:
        image = Image.open(file)
        original_size = image.size

        resized_image = image.resize(size, Image.Resampling.LANCZOS)

        logger.info(
            f"Image resized successfully "
            f"from original size: {original_size} to new size: {size}"
        )

        return resized_image
    except IOError as e:
        logger.error(f"Failed to open the image file: {e}")
        return None
    except ValueError as e:
        logger.error(f"Invalid parameters for resizing: {e}")
        return None


def encode_image_to_base64(uploaded_file: IO[bytes]) -> Optional[str]:
    """Encodes an image to a base64 string."""
    try:
        image_bytes: bytes = uploaded_file.read()
        encoded_image: str = base64.b64encode(image_bytes).decode('utf-8')
        logger.info("Image encoded successfully")
        return encoded_image
    except FileNotFoundError:
        logger.error("Image file not found")
        return None


def load_prompt() -> Optional[dict]:
    """Loads prompt configurations from a YAML file with error handling."""
    try:
        base_path = os.path.abspath(os.path.dirname(__file__))
        prompts_path = os.path.join(
            base_path, '..', '..', '..', 'prompts.yaml'
        )
        with open(prompts_path, "r", encoding='utf-8') as file:
            prompt_file = yaml.safe_load(file)

        prompt = prompt_file['chat_classification_prompt']
        logger.info("Prompt loaded successfully.")
        return prompt
    except FileNotFoundError:
        logger.error("Prompt configuration file not found.")
        return None
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        return None
    except KeyError:
        logger.error("Missing 'chat_classification_prompt' key in template.")
        return None


def api_gpt_call(image_base64: str, prompt) -> Optional[dict]:
    """Calls the OpenAI API and returns the result."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
            "model": GPT_VERSION,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }],
            "max_tokens": MAX_TOKENS
        }

        logger.info("Request body created successfully")

        response = requests.post(
            url=GPT_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()  # Raises an HTTPError for bad requests
        logger.info("OpenAI API call was successful")

        return response
    except requests.exceptions.RequestException as exc:
        logger.error(f"Failed to send request: {exc}", exc_info=True)
        return None


def extract_text_from_api_gpt_response(response: dict) -> Optional[str]:
    """Extracts and returns the full text from an OpenAI API response."""
    try:
        full_text = response.choices[0].message.content
        logger.info("Text extracted successfully")
        return full_text
    except KeyError as exc:
        logger.error(f"Key error while extracting text: {exc}")
        return None


def predict_gpt_vision_model(uploaded_file: IO[bytes]) -> Optional[str]:
    """Extracts text from an image using the OpenAI API."""
    resized_image = resize_image(uploaded_file)
    if not resized_image:
        logger.error("Failed to resize the image.")
        return None

    image_base64 = encode_image_to_base64(resized_image)
    if not image_base64:
        logger.error("Failed to encode the image.")
        return None

    prompt = load_prompt()
    if not prompt:
        logger.error("Failed to load the prompt.")
        return None

    response = api_gpt_call(image_base64, prompt)
    if not response:
        logger.error("OpenAI API call failed.")
        return None

    extracted_text = extract_text_from_api_gpt_response(response)
    if not extracted_text:
        logger.info("Failed to extract text from the OpenAI API response.")
        return None

    logger.info(f"Text classified successfully: {extracted_text}")

    return extracted_text
