from __future__ import print_function
import streamlit as st
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
from multiapp import MultiApp
from apps import home,sketch,inpaint,stadap,textonimg,Edge_Cont,Face_detect,Crop,filters,Feature_detect,abtus,img_edit
app = MultiApp()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from io import BytesIO
import base64
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import time
import copy
import os
import math

from PIL import Image

import copy

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
# st.image(images,width=224)


# option = st.selectbox(
#     'Select from the options',
#     ('Home', 'Filters', 'Doc scanner','add text'), key = 1)


# if(option=='Filters'):
#     opt = st.selectbox(
#     'Select from the options',
#     ('sepia', 'Filter1', 'filter2','filter3'), key = 2)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Clothes Transfer", abtus.app)
app.add_app("Image Editing", img_edit.app)
app.add_app("Add filters to image", filters.app)
app.add_app("Sketch", sketch.app)
app.add_app("Image inpainting", inpaint.app)
app.add_app("Image Scanner", stadap.app)
app.add_app("Add Title to image", textonimg.app)
app.add_app("Image Cropping", Crop.app)
app.add_app("Edge and Contour detection ", Edge_Cont.app)
app.add_app("Face detection", Face_detect.app)
app.add_app("Feature Detection", Feature_detect.app)

# The main app
app.run()


