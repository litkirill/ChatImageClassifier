import os
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)  # Get OpenAI API Key from environment variable

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.2,
    "max_tokens": 40,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}  # Parameters for GPT

GPT_VERSION = "gpt-3.5-turbo-0125"

CATALOG_ID = os.getenv('CATALOG_ID')
YANDEX_OCR_API_KEY = os.getenv('YANDEX_OCR_API_KEY')

YANDEX_OCR_URL = os.getenv('YANDEX_OCR_URL')