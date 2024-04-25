import streamlit as st
from PIL import Image
from src.models.easyocr_gpt_classifier import predict_model

st.title('Классификатор изображений')
uploaded_file = st.file_uploader("Загрузите изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    is_chat = predict_model(image)
    if is_сhat:
        st.write("Это скриншот переписки")
    else:
        st.write("Это НЕ скриншот переписки")
