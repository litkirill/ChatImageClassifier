import streamlit as st
import time
from typing import IO, Optional
from src.models.ocr_gpt_classifier import predict_model
from src.models.ocr_gpt_classifier import LabelType
from datetime import datetime


def process_image(uploaded_file: IO[bytes]) -> (Optional[LabelType], datetime):
    start_time = time.time()
    prediction = predict_model(uploaded_file)
    end_time = time.time()
    processing_time = end_time - start_time
    if prediction is None:
        return None, processing_time
    return prediction, processing_time


st.title('Классификатор изображений')
uploaded_file = st.file_uploader("Загрузите изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Загруженное изображение', use_column_width=True)
    if st.button('Распознать изображение'):
        prediction, processing_time = process_image(uploaded_file)
        if prediction is None:
            st.error("Ошибка при обработке изображения.")
        else:
            result_message = "Это скриншот переписки" if prediction == LabelType.CHAT else "Это НЕ скриншот переписки"
            st.success(result_message)
            st.write(f"Время обработки изображения: {processing_time:.2f} сек.")

