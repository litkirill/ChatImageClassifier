# Классификатор изображений переписок

## Описание проекта
Этот проект представляет собой приложение Streamlit, которое использует Yandex Vision OCR для распознавания текста на изображениях и с помощью ChatGPT классифицирует их как скриншоты переписок или другие типы изображений.

## Функции
- Загрузка изображений через пользовательский интерфейс.
- Использование OCR для извлечения текста из изображений.
- Классификация изображений на основе содержащегося в них текста.
- Отображение результатов классификации в реальном времени.

## Технологии
- Python 3.12.1
- Yandex Vision OCR
- OpenAI
- Streamlit
- Loguru
- Docker/Docker Compose
- Poetry для управления зависимостями

### Переменные окружения

Укажите переменные окружения в файле .env:
- OPENAI_API_KEY: ключ API для OpenAI.
- GPT_URL: URL-адрес сервиса OpenAI
- CATALOG_ID: Идентификатор каталога Yandex Cloud.
- YANDEX_OCR_API_KEY: ключ API для Yandex Vision OCR.
- YANDEX_OCR_URL: URL-адрес для Yandex Vision OCR

## Установка и Запуск
```bash
git clone https://github.com/litkirill/ChatImageClassifier.git
cd ChatImageClassifier
docker-compose up --build