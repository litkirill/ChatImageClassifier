# Классификатор изображений переписок

## Описание проекта
Этот проект представляет собой приложение Streamlit, которое использует Yandex Vision OCR для распознавания текста на изображениях и с помощью YandexGPT Pro классифицирует их как скриншоты переписок или другие типы изображений.

## Функции
- Загрузка изображений через пользовательский интерфейс.
- Использование OCR для извлечения текста из изображений.
- Классификация изображений на основе содержащегося в них текста.
- Отображение результатов классификации в реальном времени.

## Технологии
- Python 3.12
- Yandex Vision OCR
- YandexGPT
- Streamlit
- Docker/Docker Compose
- Poetry для управления зависимостями

### Переменные окружения

Укажите переменные окружения в файле .env:
- CATALOG_ID: Идентификатор каталога Yandex Cloud.
- YANDEX_GPT_API_KEY: API ключ для доступа к YandexGPT.
- YANDEX_OCR_API_KEY: API ключ для доступа к Yandex Vision OCR.

## Установка и Запуск
```bash
git clone https://github.com/litkirill/ChatImageClassifier.git
cd ChatImageClassifier
docker-compose up --build