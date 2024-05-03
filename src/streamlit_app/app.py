"""Module for an image classification Streamlit app."""

import os
import time
from typing import IO, Optional
import streamlit as st
from src.config import logger, REQUIRED_ENV_VARS
from src.models.ocr_gpt_classifier import predict_model
from src.models.ocr_gpt_classifier import LabelType


def check_env_vars(env_vars):
    """Checks if all required environment variables are present and set. If
    any are missing, logs an error, informs the user via the Streamlit
    interface, and halts the application."""
    missing_vars = [var for var in env_vars if not os.getenv(var)]
    if missing_vars:
        error_message = f"Missing critical environment variables: {', '.join(missing_vars)}"
        logger.error(error_message)
        st.error(
            "Не удалось инициализировать приложение из-за отсутствия "
            "необходимых конфигураций. Пожалуйста, обратитесь в техническую "
            "поддержку."
        )
        st.stop()  # Stop further execution
    else:
        logger.info(
            "All required environment variables are loaded successfully."
        )


def process_image(uploaded_file: IO[bytes]) -> (Optional[LabelType], float):
    """Process the image file to predict its label and calculate processing time."""
    logger.info("Starting image processing.")
    start_time = time.time()

    image_prediction = predict_model(uploaded_file)
    if prediction is None:
        logger.error("Error processing image")
        return None, 0

    end_time = time.time()
    image_processing_time = end_time - start_time
    logger.info(f"Image processed in {processing_time:.2f} seconds.")

    return image_prediction, image_processing_time


logger.info("Initializing the application...")

check_env_vars(REQUIRED_ENV_VARS)

st.title('Классификатор изображений')
file_to_upload = st.file_uploader(
    "Загрузите изображение",
    type=['png', 'jpg', 'jpeg']
)

if file_to_upload is not None:
    st.image(
        file_to_upload,
        caption='Загруженное изображение',
        use_column_width=True
    )
    if st.button('Распознать изображение'):
        prediction, processing_time = process_image(file_to_upload)
        if prediction is None:
            st.error("Ошибка при обработке изображения.")
        else:
            RESULT_MESSAGE = \
                "Это скриншот переписки" if prediction.value else "Это НЕ скриншот переписки"
            logger.info(f"Prediction result: {RESULT_MESSAGE}")
            st.success(RESULT_MESSAGE)
            st.write(
                f"Время обработки изображения: {processing_time:.2f} сек."
            )
