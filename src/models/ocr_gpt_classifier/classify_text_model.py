import os
import yaml
from typing import Optional
from loguru import logger
from openai import OpenAI
from openai import OpenAIError
from src.config import OPENAI_API_KEY, OPENAI_COMPLETION_OPTIONS, \
    GPT_PROXY_URL, GPT_VERSION
from .types import LabelType

if not OPENAI_API_KEY:
    logger.error(
        "Critical environment variables are missing. "
        "Please check OPENAI_API_KEY"
    )
    raise EnvironmentError(
        "Critical environment variables are missing. "
        "Please check OPENAI_API_KEY"
    )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=GPT_PROXY_URL
)

logger.add(
    "logs/logs_from_app.log", format="{time} {level} {message}",
    level="DEBUG", rotation="10 MB", compression="zip"
)


def load_prompt_template() -> dict:
    """Loads prompt configurations from a YAML file with error handling."""
    try:
        base_path = os.path.abspath(os.path.dirname(__file__))
        prompts_path = os.path.join(
            base_path, '..', '..', '..', 'prompts.yaml'
        )
        with open(prompts_path, "r", encoding='utf-8') as file:
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
        return prompt_text
    except KeyError:
        logger.error("Missing 'chat_classification_prompt' key in template.")
        raise KeyError("Missing 'chat_classification_prompt' key in template.")


def api_call(prompt: str) -> Optional[dict]:
    """Sends a request to the API and returns the response with error
    handling."""
    try:
        logger.info("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model=GPT_VERSION,
            messages=[{"role": "user", "content": prompt}],
            **OPENAI_COMPLETION_OPTIONS,
        )

        return response
    except OpenAIError as e:
        logger.error(f"An error occurred with the OpenAI API: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during text classification: {e}")
        return None


def extract_prediction(response) -> LabelType:
    """Extracts the prediction from the API response with error handling."""
    try:
        predicted_text = response.choices[0].message.content
        logger.debug("Prediction extracted successfully.")
        return predicted_text
    except KeyError as e:
        logger.error(f"Failed to extract prediction from response: {e}")
        raise KeyError(f"Failed to extract prediction from response: {e}")


def classify_text(text: str) -> Optional[LabelType]:
    """Classifies text using the OpenAI API with error handling and logging."""
    try:
        prompt = create_prompt(text)
        response = api_call(prompt)
        predicted_text = extract_prediction(response)
        logger.info(f"Text classified successfully: {predicted_text}")
        return predicted_text
    except Exception as e:
        logger.error(f"An error occurred during text classification: {e}")
        return None
