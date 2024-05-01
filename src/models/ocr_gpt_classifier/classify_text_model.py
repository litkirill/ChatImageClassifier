import os
import yaml
from typing import Optional
from openai import OpenAI
from openai import OpenAIError
from .types import LabelType
from src.config import logger, \
    OPENAI_API_KEY, OPENAI_COMPLETION_OPTIONS, GPT_PROXY_URL, GPT_VERSION

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=GPT_PROXY_URL
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
        logger.info("Prompt template loaded successfully.")
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
        logger.info("Prompt created successfully.")
        return prompt_text
    except KeyError:
        logger.error("Missing 'chat_classification_prompt' key in template.")
        raise KeyError("Missing 'chat_classification_prompt' key in template.")


def api_call(prompt: str) -> Optional[dict]:
    """Call the OpenAI API to get a response."""
    try:
        response = client.chat.completions.create(
            model=GPT_VERSION,
            messages=[{"role": "user", "content": prompt}],
            **OPENAI_COMPLETION_OPTIONS,
        )
        logger.info("OpenAI API call was successful")
        return response
    except OpenAIError as e:
        logger.error(f"An error occurred with the OpenAI API: {e}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during the OpenAI API call: {e}")
        return None


def extract_prediction(response) -> LabelType:
    """Extracts the prediction from the API response with error handling."""
    try:
        predicted_text = response.choices[0].message.content
        logger.info("Prediction extracted successfully.")
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
