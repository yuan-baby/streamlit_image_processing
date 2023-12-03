import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.title("Morphological Processing:smile:")
st.sidebar.markdown("# Morphological Processing")

'''
The functions of this page are listed as below:
'''
st.markdown(
    '''
    - Find the edge of the image with mophological methods
    - Enhance the image (with gray level morphological methods)
    - Segment the image with mophological methods
    '''
)

# Load images
upload_file = st.file_uploader("choose a picture and watch miracle happen!:sunglasses:")

data_type = st.selectbox(
    'What is the data type of your file?',
    ('uint16','uint8','float32')
)

if (upload_file is not None) and (data_type is not None):
    data = upload_file.getvalue()
    if data_type == 'uint8':
        data = np.frombuffer(data, dtype=np.uint8)
    elif data_type == 'uint16':
        data = np.frombuffer(data, dtype=np.uint16)
    elif data_type == 'float32':
        data = np.frombuffer(data, dtype=np.float32)

width = st.number_input("What's the width of your file?",min_value=0,max_value=1000,value=0,step=1)
height = st.number_input("What's the height of your file?",min_value=0,max_value=1000,value=0,step=1)

st.markdown('You can press the `show original image!` button only after uploaded your image and set the parameters')


if st.button('show original image!') and (upload_file is not None) and (width != 0) and (height != 0):
    # 从缓冲读取数据，故每次使用data时都需要进行下面两个操作
    data = data.reshape((height,width))
    data = np.clip(data, 0, 255)
    st.image(data, caption='original image')

st.markdown("## Boundary extraction")

def BoundaryExtraction(data, size):
    data = data.reshape((height,width))
    data = np.clip(data, 0, 255)
    element_size = (size,size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, element_size)
    img_erode = cv2.erode(data, kernel)
    result = data - img_erode
    return result

structure_element_size_be = st.number_input("What's the size of your structure element?",min_value=0,max_value=1000,value=0,step=1,key='be')

st.markdown('You can press the `extract the boundary!` button only after uploaded your image and set the parameters')

if st.button('extract the boundary!') and (structure_element_size_be != 0):
    boundary_image = BoundaryExtraction(data, size = structure_element_size_be)
    st.image(boundary_image, caption='boundary')

st.markdown("## Enhance the image")

def enhance_image(data, kernel_size):
    # Normalize the image to 0-255 range and convert it to 8-bit grayscale
    image = data.reshape((height,width))
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create a circular structuring element
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform morphological opening operation first and then closing operation
    image_opened = cv2.morphologyEx(image_normalized, cv2.MORPH_OPEN, circle_kernel)
    image_enhanced = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, circle_kernel)

    return image_enhanced

structure_element_size_enhance = st.number_input("What's the size of your structure element?",min_value=0,max_value=1000,value=0,step=1,key='enhance')

st.markdown('You can press the `enhance the image!` button only after uploaded your image and set the parameters')

if st.button('enhance the image!') and (structure_element_size_enhance != 0):
    boundary_image = enhance_image(data, kernel_size = structure_element_size_enhance)
    st.image(boundary_image, caption='enhanced')

st.markdown("## Segment the image")

st.markdown("### Canny Detection")

def CannyEdge(data,lower_threshold, upper_threshold):
    data = data.reshape((height,width))
    edge = cv2.Canny(data.astype(np.uint8), lower_threshold, upper_threshold)
    return edge

st.markdown('Input the thresholds manually to pick out the proper thresholds for edge detection, which can be used for thresholding.You can press the `Canny detect!` button only after uploaded your image and set the parameters')

lower_threshold = st.number_input("What's the lower_threshold?",min_value=0,max_value=1000,value=0,step=1)
upper_threshold = st.number_input("What's the upper_threshold?",min_value=0,max_value=1000,value=0,step=1)

if st.button('Canny detect!') and (lower_threshold != 0) and (upper_threshold != 0):
    edge_image = CannyEdge(data,lower_threshold, upper_threshold)
    st.image(edge_image, caption='Canny-edge')

st.markdown("### Thresholding")

st.markdown('Input the thresholds manually to do thresholding as the final step of segmentation.You can press the `Threshold!` button only after Canny-detection and set the parameters')
threshold = st.number_input("Input a random threshold",min_value=0,max_value=1000,value=0,step=1)

if st.button('Threshold!') and (threshold != 0):
    edge_image = CannyEdge(data,lower_threshold, upper_threshold)
    _, thresholded = cv2.threshold(edge_image, threshold, 255, cv2.THRESH_BINARY)
    st.image(thresholded, caption='thresholded')