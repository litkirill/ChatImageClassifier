import streamlit as st
from src.models.ocr_gpt_classifier import predict_model
from src.models.ocr_gpt_classifier import LabelType
from datetime import datetime
import time


def process_image(uploaded_file: str) -> (LabelType, datetime):
    start_time = time.time()
    prediction = predict_model(uploaded_file)
    end_time = time.time()
    processing_time = end_time - start_time
    return prediction, processing_time


st.title('Классификатор изображений')
uploaded_file = st.file_uploader("Загрузите изображение",
                                 type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Загруженное изображение', use_column_width=True)
    result = st.button('Распознать изображение')

    if result:
        predict, processing_time = process_image(uploaded_file)
        if predict.value:
            st.write("Это скриншот переписки")
        else:
            st.write("Это НЕ скриншот переписки")

        st.write(f"Время обработки изображения: {processing_time:.2f} сек.")
