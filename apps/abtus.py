import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import base64

def app():
    st.title("Neural Style Transfer for Wedding Photo")
    image = Image.open('imgs/lake.jpeg')
    st.image(image, caption='Clothes Transfer!', use_column_width=True)
    st.header('Clothes transfer')
    st.write('If you wish to perform clothes transfer, please kindly visit the URL below, upload your images to Google Colab, press (Ctrl+F9), wait for the results and download the output!')
    st.write("check out this [link](https://colab.research.google.com/drive/1WyZGHuWLHvYbVFH6xThtqHecZ15TxsPT?usp=sharing)")
    
    st.write("")
    st.write("")
    st.image(Image.open("wedding photo.jpg"), caption='Example of Content Image.',width=224)
    st.image(Image.open("style.jpg"), caption='Example of Style Image.',width=224)
    st.image(Image.open("dress_shape.jpg"), caption='Example of Masked Image.',width=224)
    st.image(Image.open("final_output.jpg"), caption='Example of Output Image.',width=224)
    
    st.write("")
    st.write("")
    st.subheader("GUIDANCE")
    st.markdown("![BEFORE](https://media.giphy.com/media/DIqXdtz1VimVaox2pL/giphy.gif)")
    st.markdown("![AFTER](https://media.giphy.com/media/wMRrb4zLcIs2yBs8dE/giphy.gif)")
