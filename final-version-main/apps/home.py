import streamlit as st
from PIL import Image
def app():
    image = Image.open('imgs/lake.jpeg')
    st.image(image, caption='Welcome to my webapp!', use_column_width=True)
    st.subheader('Lew Jun Xian 18ACB04184')
    st.subheader("Computer Science", anchor=None)
