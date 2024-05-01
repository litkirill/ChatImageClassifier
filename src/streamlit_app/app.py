import os
import time
import streamlit as st
from typing import IO, Optional
from src.config import logger, REQUIRED_ENV_VARS
from src.models.ocr_gpt_classifier import predict_model
from src.models.ocr_gpt_classifier import LabelType


def check_env_vars(vars):
    """Checks if all required environment variables are present and set. If
    any are missing, logs an error, informs the user via the Streamlit
    interface, and halts the application."""
    missing_vars = [var for var in vars if not os.getenv(var)]
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
    logger.info("Starting image processing.")
    start_time = time.time()

    try:
        prediction = predict_model(uploaded_file)
        if prediction is None:
            logger.warning(
                "No prediction could be made for the uploaded image."
            )
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        st.error("Ошибка при обработке изображения.")
        return None, 0

    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Image processed in {processing_time:.2f} seconds.")

    return prediction, processing_time


logger.info("Initializing the application...")

check_env_vars(REQUIRED_ENV_VARS)

st.title('Классификатор изображений')
uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption='Загруженное изображение',
        use_column_width=True
    )
    if st.button('Распознать изображение'):
        prediction, processing_time = process_image(uploaded_file)
        if prediction is None:
            st.error("Ошибка при обработке изображения.")
        else:
            result_message = "Это скриншот переписки" if prediction == LabelType.CHAT else "Это НЕ скриншот переписки"
            logger.info(f"Prediction result: {result_message}")
            st.success(result_message)
            st.write(
                f"Время обработки изображения: {processing_time:.2f} сек."
            )
