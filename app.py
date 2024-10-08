import streamlit as st
import gdown
import pandas as pd
import numpy as np
from PIL import Image, ImageColor
import os
from ultralytics import YOLO


# Import packages
from pathlib import Path
import PIL
import pandas as pd
import streamlit as st

# Setting page layout
st.set_page_config(
    page_title="Dyslexic Handwriting Correction Tool",
    page_icon="ðŸ¤–",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Dyslexic Handwriting Correction Tool")

# Sidebar
st.sidebar.header("Image Upload")









from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'video_1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
}









# Function to download the file from Google Drive
@st.cache_resource()
def download_file_from_drive(url, output):
    gdown.download(url, output, quiet=False)



# Google Drive file link (shared link)
file_url = 'https://drive.google.com/uc?id=15s8ScVQW8wGDBzG4QlTXXctbpXiSuVnb'
output_file = 'model.pt'

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
                model = YOLO(output_file)
                st.success('Model loaded successfully!')
                #st.write(model.summary())
        except OSError as e:
            st.error(f"Error loading model: {e}")
    else:
        st.error('Downloaded file is empty or corrupted.')
else:
    st.error('Error: Model file not found.')





# Model Options
model_type = "Detection"

confidence = 0.001

# set list of class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Read the class names and their correct positions
dyslexic_letters_df = pd.read_csv('dyslexic_letters.csv')

# Extract the 'Class' column as keys and 'Position' as values in a dictionary
class_position_dict = dict(zip(dyslexic_letters_df['Class'], dyslexic_letters_df['Position']))

# Selecting Detection Or Segmentation
#model_path = Path(settings.DETECTION_MODEL)

source_img = None

source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

col1, col2 = st.columns(2)

with col1:
    try:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption="Default Image",
                     use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Uploaded Image",
                     width=80)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    if source_img is None:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(
            default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image',
                 use_column_width=True)
    else:
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            # st.image(res_plotted, caption='Detected Image',
            #         use_column_width=True)

            try:
                st.write("Detection Results")

                highest_prob_class = None
                highest_prob = 0.0
                position = 1
                predictions_dict = {}

                for box in boxes:
                    box_data = box.data.cpu().numpy().tolist()  # Convert box data to list
                    class_id = int(box_data[0][-1])  # Assuming class ID is the last entry
                    probability = box_data[0][-2]  # Assuming probability/confidence is the second last entry

                    # Get the class name from the class ID
                    class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

                    # Update highest probability class
                    if probability > highest_prob:
                        highest_prob = probability
                        highest_prob_class = class_name

                    predictions_dict[f"{class_name}"] = position
                    st.write(f"Predicted letter: {class_name} at Position {position}")

                    position += 1

                # Check if the predicted letter and its position match the reference values
                for key, value in class_position_dict.items():
                    if key in predictions_dict and predictions_dict[key] == value:
                        st.write(f"Predicted letter: {key}")
                        break;


            except Exception as ex:
                st.write("No image is uploaded yet!")
                st.error(ex)

