import streamlit as st

# The function of main page is to show the function 
# and characteristics of the web app 
st.title("Main Page:pray:")
st.sidebar.markdown("# Main Page")

# Introduction
st.markdown("# Introduction")
welcome = '''
Hello guys!:wave: This is an web-app for image processing and image analysis.\n
- For image zooming and shrinking, please navigate to basic operation.\n
- For image enhancement, such as contrast adjustment, please navigate to image enhancement.\n
- For morphological processing, please navigate to mophological processing\n
'''

st.markdown(welcome)