# prediction.py
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model once (global scope)
MODEL_PATH = r'D:\Study\programing\Ai\a\Potato-Disease-Classification-using-Deep-Learning-main\model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def prediction(image, class_names=None):
    if class_names is None:
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    # Preprocess the image
    img = Image.open(image)
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)

    return {
        "predict_class": predicted_class,
        "predict_presntage": confidence
    }
