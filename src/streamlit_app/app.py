import streamlit as st
from PIL import Image
from src.models.ocr_gpt_classifier import predict_model
from src.models.ocr_gpt_classifier import LabelType
from datetime import datetime
import time


def process_image(image: Image.Image) -> (LabelType, datetime):
    start_time = time.time()
    prediction = predict_model(image)
    end_time = time.time()
    processing_time = end_time - start_time
    return prediction, processing_time


st.title('Классификатор изображений')
uploaded_file = st.file_uploader("Загрузите изображение",
                                 type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    result = st.button('Распознать изображение')

    if result:
        predict, processing_time = process_image(image)
        if predict.value:
            st.write("Это скриншот переписки")
        else:
            st.write("Это НЕ скриншот переписки")

        st.write(f"Время обработки изображения: {processing_time:.2f} сек.")
