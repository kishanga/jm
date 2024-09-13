import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image, ImageColor
import os
import tensorflow as tf


# Function to download the file from Google Drive
@st.cache_resource()
def download_file_from_drive(url, output):
    gdown.download(url, output, quiet=False)



# Google Drive file link (shared link)
file_url = 'https://drive.google.com/uc?id=15s8ScVQW8wGDBzG4QlTXXctbpXiSuVnb'
output_file = 'model.keras'

# Streamlit app starts here
st.title("Predictor")

# Button to trigger model download
#if st.button('Download and Load Model'):
with st.spinner('Model loading. Please wait ...'):

    
    download_file_from_drive(file_url, output_file)
    st.success('Model downloaded successfully!')

# Check if the file exists
if os.path.exists(output_file):
    # Check if the file is indeed an .h5 file by size (usually > few MBs)
    file_size = os.path.getsize(output_file)
    #st.write(f"Downloaded file size: {file_size / (1024 * 1024)} MB")
    
    if file_size > 0:  # Ensuring the file is not empty
        try:
            with st.spinner('Loading model...'):
                model = load_model(output_file)
                st.success('Model loaded successfully!')
                #st.write(model.summary())
        except OSError as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error('Downloaded file is empty or corrupted.')
else:
    st.error('Error: Model file not found.')
