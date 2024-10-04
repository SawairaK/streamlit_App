import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.metrics import MeanAbsoluteError

# Load the trained model
model = tf.keras.models.load_model('Age_and_Gender_detection.h5', 
                                   custom_objects={'mae': MeanAbsoluteError()})  # Use the correct custom object

# Recompile the model with the required settings
model.compile(loss=['binary_crossentropy', MeanAbsoluteError()], 
              optimizer='adam', 
              metrics={'gender_out': 'accuracy', 'age_out': MeanAbsoluteError()})

# Gender mapping
gender_mapping = {1: 'Female', 0: 'Male'}

# Define Streamlit app
st.title("Age and Gender Prediction App")

st.write("This app predicts the age and gender of a person based on an uploaded image.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Prediction button
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing the uploaded image
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to match model input
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.reshape(1, 128, 128, 1)  # Reshape for the model

    # Predict button
    if st.button('Predict'):
        pred = model.predict(img_array)
        pred_gender = gender_mapping[round(pred[0][0][0])]
        pred_age = round(pred[1][0][0])
        
        st.write(f"Predicted Gender: {pred_gender}")
        st.write(f"Predicted Age: {pred_age}")
