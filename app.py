import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Load the model ---
# Use a caching decorator to load the model only once.
# This makes the app run much faster.
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_detection_model.keras')
    return model

model = load_model()

# Define the labels for the output
class_names = ['Normal', 'Pneumonia']

# --- Set up the Streamlit app layout ---
st.title('Pneumonia Detection from Chest X-ray')
st.markdown('A deep learning application to classify chest X-ray images.')

st.header('Upload an X-ray Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to a NumPy array for model prediction
    img_array = np.array(image)

    # Preprocess the image to match the model's input requirements
    img_array = tf.image.resize(img_array, (150, 150))
    img_array = np.expand_dims(img_array, axis=0) # Add a batch dimension
    img_array = img_array / 255.0 # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction) # Get the index of the predicted class
    confidence = np.max(prediction) # Get the confidence score

    # Display the prediction and confidence
    st.header('Prediction')
    if class_names[prediction_class] == 'Pneumonia':
        st.error(f'Prediction: {class_names[prediction_class]}')
        st.write(f'Confidence: {confidence:.2f}')
        st.markdown("⚠️ **Warning:** This prediction suggests the presence of pneumonia. A medical professional's diagnosis is required for confirmation.")
    else:
        st.success(f'Prediction: {class_names[prediction_class]}')
        st.write(f'Confidence: {confidence:.2f}')
        st.markdown("✅ **Great!** This prediction suggests no sign of pneumonia. Always consult a medical professional for a definitive diagnosis.")

    st.markdown("---")
    st.markdown("### How to interpret the results:")
    st.markdown("- **Normal:** The model predicts no signs of pneumonia.")
    st.markdown("- **Pneumonia:** The model predicts the presence of pneumonia.")
    st.markdown("The confidence score indicates how certain the model is about its prediction. A score closer to 1.0 means higher certainty.")