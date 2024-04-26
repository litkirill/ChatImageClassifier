import os
import easyocr
from PIL import Image
from dotenv import load_dotenv
import requests
import yaml
from enum import Enum

load_dotenv()
CATALOG_ID = os.getenv('CATALOG_ID')
API_KEY = os.getenv('YANDEX_API_KEY')

reader = easyocr.Reader(['ru', 'en'], gpu=False)


class LabelType(Enum):
    CHAT = 1
    NOT_CHAT = 0


def load_prompt_template() -> str:
    base_path = os.path.abspath(os.path.dirname(__file__))
    prompts_path = os.path.join(base_path, '..', '..', 'prompts.yaml')
    with open(prompts_path, "r") as file:
        prompts = yaml.safe_load(file)
    return prompts


def create_prompt(text: str) -> dict:
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


def extract_text(image: Image.Image) -> str:
    results = reader.readtext(image, detail=0, paragraph=True)
    return " ".join([result for result in results])


def classify_text(text: str) -> str:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {API_KEY}"
    }

    prompt = create_prompt(text)

    response = requests.post(url, headers=headers, json=prompt)
    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")

    response_dict = response.json()
    predicted = response_dict['result']['alternatives'][0]['message'][
        'text']
    return predicted


def predict_model(image: Image.Image) -> LabelType:
    text = extract_text(image)
    predicted = classify_text(text)

    return LabelType.CHAT if "<chat>" in predicted else LabelType.NOT_CHAT