import os
import requests
import yaml
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
from .types import LabelType

load_dotenv()

CATALOG_ID = os.getenv('CATALOG_ID')
YANDEX_GPT_API_KEY = os.getenv('YANDEX_GPT_API_KEY')

if not CATALOG_ID or not YANDEX_GPT_API_KEY:
    logger.error(
        "Critical environment variables are missing. Please check CATALOG_ID "
        "and YANDEX_GPT_API_KEY.")
    raise EnvironmentError(
        "Critical environment variables are missing. Please check CATALOG_ID "
        "and YANDEX_GPT_API_KEY.")

url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

logger.add(
    "logs/logs_from_app.log", format="{time} {level} {message}",
    level="DEBUG", rotation="10 MB", compression="zip"
)


def load_prompt_template() -> str:
    """Loads prompt configurations from a YAML file with error handling."""
    try:
        base_path = os.path.abspath(os.path.dirname(__file__))
        prompts_path = os.path.join(base_path, '..', '..', '..',
                                    'prompts.yaml')
        with open(prompts_path, "r") as file:
            prompts = yaml.safe_load(file)
        logger.debug("Prompt template loaded successfully.")
        return prompts
    except FileNotFoundError:
        logger.error("Prompt configuration file not found.")
        raise FileNotFoundError("Prompt configuration file not found.")
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file: {exc}")
        raise RuntimeError(f"Error parsing YAML file: {exc}")


def create_prompt(text: str) -> dict:
    """Formats a prompt for an API request using the specified text with
    error handling."""
    try:
        template = load_prompt_template()
        prompt_text = template['chat_classification_prompt'].format(
            message_text=text)
        logger.debug("Prompt created successfully.")
    except KeyError:
        logger.error("Missing 'chat_classification_prompt' key in template.")
        raise KeyError("Missing 'chat_classification_prompt' key in template.")
    return {
        "modelUri": f"gpt://{CATALOG_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.,
            "maxTokens": 100
        },
        "messages": [{"role": "system", "text": prompt_text}]
    }


def prepare_request(text: str) -> dict:
    """Prepares the data for the API request."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_GPT_API_KEY}"
        }
        prompt = create_prompt(text)
        logger.debug("Request has been prepared successfully.")
        return {
            "url": url,
            "headers": headers,
            "data": prompt
        }
    except Exception as e:
        logger.error(f"Failed to prepare the request: {e}")
        raise Exception(f"Failed to prepare the request: {e}")


def send_request(request_details: dict) -> dict:
    """Sends a request to the API and returns the response with error
    handling."""
    try:
        response = requests.post(request_details["url"],
                                 headers=request_details["headers"],
                                 json=request_details["data"])
        response.raise_for_status()  # Will raise an HTTPError for bad HTTP status codes
        logger.debug("API call was successful.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send request: {e}")
        raise ConnectionError(f"Failed to send request: {e}")


def extract_prediction(response_dict: dict) -> LabelType:
    """Extracts the prediction from the API response with error handling."""
    try:
        predicted_text = response_dict['result']['alternatives'][0]['message'][
            'text']
        logger.debug("Prediction extracted successfully.")
        return predicted_text
    except KeyError as e:
        logger.error(f"Failed to extract prediction from response: {e}")
        raise KeyError(f"Failed to extract prediction from response: {e}")


def classify_text(text: str) -> Optional[LabelType]:
    """Classifies text using the Yandex API with error handling and logging."""
    try:
        request_details = prepare_request(text)
        response_dict = send_request(request_details)
        predicted_text = extract_prediction(response_dict)
        logger.info(f"Text classified successfully: {predicted_text}")
        return predicted_text
    except Exception as e:
        logger.error(f"An error occurred during text classification: {e}")
        return None
