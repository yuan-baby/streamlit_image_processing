import streamlit as st
import numpy as np
import os
import cv2 
import pydicom
from PIL import Image


st.title("Basic Operation:relaxed:")
st.sidebar.markdown("# Basic Operation")

'''
The functions of this page are listed as below:
'''
st.markdown(
    '''
    - Reducing the Number of Intensity Levels in an Image.
    - Zooming and Shrinking Images by Pixel Replication.
    - Zooming and Shrinking Images by Bilinear Interpolation.
    '''
)

# functions
# Reducing the Number of Intensity Levels in an Image.
def reduce_intensity_levels(image, num_levels):
    output_image = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
    max_value = np.amax(image)
    min_value = np.amin(image)
    intensity_range = max_value - min_value
    interval = intensity_range / (num_levels - 1)

    for i in range(len(output_image)):
        for j in range(len(output_image[i])):
            intensity_level = int(round((image[i][j] - min_value) / interval)) * interval + min_value
            output_image[i][j] = intensity_level

    return output_image

# Zooming and Shrinking Images by Pixel Replication.
def pixel_replication(image, zoom):
    output_width = int(image.shape[1] * zoom)
    output_height = int(image.shape[0] * zoom)
    output_image = np.zeros((output_height, output_width), dtype=np.uint8)

    for i in range(output_height):
        for j in range(output_width):
            input_x = int(j / zoom)
            input_y = int(i / zoom)
            output_image[i][j] = image[input_y][input_x]

    return output_image

# Zooming and Shrinking Images by Bilinear Interpolation.
def bilinear_interpolation(image, zoom):
    output_width = int(image.shape[1] * zoom)
    output_height = int(image.shape[0] * zoom)
    output_image = np.zeros((output_height, output_width), dtype=np.uint8)

    for i in range(output_height):
        for j in range(output_width):
            input_x = j / zoom
            input_y = i / zoom

            x1 = int(input_x)
            y1 = int(input_y)
            x2 = min(x1 + 1, image.shape[1] - 1)
            y2 = min(y1 + 1, image.shape[0] - 1)

            dx = input_x - x1
            dy = input_y - y1

            pixel_value = (1 - dx) * (1 - dy) * image[y1][x1] + dx * (1 - dy) * image[y1][x2] + \
                          (1 - dx) * dy * image[y2][x1] + dx * dy * image[y2][x2]

            output_image[i][j] = int(pixel_value)

    return output_image
    

# Load images
upload_file = st.file_uploader("choose a picture and watch miracle happen!:sunglasses:",type=['jpg','png','jpeg','dcm'])

option = st.selectbox(
    'What is the type of your file?',
    ('jpg','jpeg','png','dicom')
)

if upload_file is not None:
    if option == 'dicom':
        dicom_data = pydicom.dcmread(upload_file)
        pixel_data = dicom_data.pixel_array
        # convert to grey image
        bits_allocated = dicom_data.BitsAllocated
        # 根据位深度进行灰度化
        if bits_allocated == 8:  # 8位深度
            image = cv2.normalize(pixel_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif bits_allocated == 16:  # 16位深度
            image = cv2.normalize(pixel_data, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
            image = cv2.convertScaleAbs(image // 256)  # 将16位图像转换为8位图像
        else:
            raise ValueError("Unsupported bit depth")
        st.image(image, caption='your pic')
    else:
        image = Image.open(upload_file)
        st.image(image, caption='your pic')

# Process
if st.button('reduce the pic!'):
    reduced_image = reduce_intensity_levels(image, num_levels=4)
    st.image(reduced_image, caption='reduced image')

if st.button('zoom the pic!(with pixel replication)'):
    zoomed_image_pr = pixel_replication(image, zoom=2)
    st.image(zoomed_image_pr, caption='zoomed image')

if st.button('shrink the pic!(with pixel replication)'):
    shrunk_image_pr = pixel_replication(image, zoom=0.5)
    st.image(shrunk_image_pr, caption='shrunk image')

if st.button('zoom the pic!(with bilinear interpolation)'):
    zoomed_image_bi = bilinear_interpolation(image, zoom=2)
    st.image(zoomed_image_bi, caption='zoomed image')

if st.button('shrink the pic!(with bilinear interpolation)'):
    shrunk_image_bi = bilinear_interpolation(image, zoom=0.5)
    st.image(shrunk_image_bi, caption='shrunk image')