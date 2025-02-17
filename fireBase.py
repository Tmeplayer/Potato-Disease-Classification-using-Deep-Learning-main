import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
import firebase_admin
from firebase_admin import credentials, storage
from io import BytesIO

filterwarnings('ignore')

if not firebase_admin._apps:
    cred = credentials.Certificate(r'D:\Study\programing\Ai\a\Potato-Disease-Classification-using-Deep-Learning-main\greenhouse-data-1-firebase-adminsdk-clq7a-27052852ea.json')  # Replace with your Firebase credentials file path
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'greenhouse-data-1.appspot.com'  
    })

def streamlit_config():
    st.set_page_config(page_title='Classification', layout='centered')

    page_background_color = """
    <style>
    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    st.markdown(f'<h1 style="text-align: center;">Potato Disease Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)


# Streamlit Configuration Setup
streamlit_config()


def download_image_from_firebase(image_path):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(image_path)

        if not blob.exists():
            raise ValueError(f"The image with path '{image_path}' does not exist in Firebase Storage.")
        
        image_data = blob.download_as_bytes()

        image = Image.open(BytesIO(image_data))

        return image
    except Exception as e:
        st.error(f"Error downloading or processing image: {str(e)}")
        return None


def prediction(image, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    img = Image.open(image)
    
    img_resized = img.resize((256, 256))
    
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    model = tf.keras.models.load_model(r'D:\Study\programing\Ai\a\Potato-Disease-Classification-using-Deep-Learning-main\model.h5')
    
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    add_vertical_space(1)
    st.markdown(f'<h4 style="color: orange;">Predicted Class : {predicted_class}<br>Confidence : {confidence}%</h4>', 
                unsafe_allow_html=True)
    
    add_vertical_space(1)
    st.image(img.resize((400, 300)))


col1, col2, col3 = st.columns([0.1, 0.9, 0.1])

with col2:
    image_path = st.text_input('Enter the image path from Firebase Storage')

if image_path:
    col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
    with col2:
        input_image = download_image_from_firebase(image_path)
        
        if input_image is not None:
            prediction(input_image)
