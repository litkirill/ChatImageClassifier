import os
import requests
import yaml
from dotenv import load_dotenv
from .types import LabelType

load_dotenv()

CATALOG_ID = os.getenv('CATALOG_ID')
YANDEX_GPT_API_KEY = os.getenv('YANDEX_GPT_API_KEY')

url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


def load_prompt_template() -> str:
    """Loads prompt configurations from a YAML file."""

    base_path = os.path.abspath(os.path.dirname(__file__))
    prompts_path = os.path.join(base_path, '..', '..', '..', 'prompts.yaml')
    with open(prompts_path, "r") as file:
        prompts = yaml.safe_load(file)

    return prompts


def create_prompt(text: str) -> dict:
    """Formats a prompt for an API request using the specified text."""

    template = load_prompt_template()
    prompt_text = template['chat_classification_prompt'].format(
        message_text=text)

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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_GPT_API_KEY}"
    }
    prompt = create_prompt(text)

    return {
        "url": url,
        "headers": headers, "data": prompt}


def send_request(request_details: dict) -> dict:
    """Sends a request to the API and returns the response."""

    response = requests.post(request_details["url"],
                             headers=request_details["headers"],
                             json=request_details["data"])
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")
    return response.json()


def extract_prediction(response_dict: dict) -> LabelType:
    """Extracts the prediction from the API response."""
    predicted_text = response_dict['result']['alternatives'][0]['message'][
        'text']

    return predicted_text


def classify_text(text: str) -> LabelType:
    """Classifies text using the Yandex API."""

    request_details = prepare_request(text)
    response_dict = send_request(request_details)
    predicted_text = extract_prediction(response_dict)

    return predicted_text
