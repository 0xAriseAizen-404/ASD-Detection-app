import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image  # Using keras.preprocessing.image instead of autism_model
from PIL import Image
import streamlit as st

# Define folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

# Preprocessing function adapted from Flask
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Match Flask target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize as in Flask
    return img_array

# Streamlit UI for image analysis
def image_analysis_ui():
    st.title("Image Analysis for Autism Detection")

    # Load the pre-trained model
    model_path = os.path.join(MODELS_FOLDER, "autism_model.h5")  # Updated model name
    if not os.path.exists(model_path):
        st.error(f"Model file autism_model.h5 not found in {MODELS_FOLDER}! Please ensure the model file is in the models folder.")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model autism_model.h5: {str(e)}")
        return

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])

    # CSS styles for prediction output (retained from your Streamlit code)
    yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
    no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'

    # Process the uploaded image
    if uploaded_file:
        try:
            # Save the uploaded file temporarily
            temp_dir = os.path.join(BASE_DIR, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            img_path = os.path.join(temp_dir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and display the uploaded image
            image_display = Image.open(img_path)
            # st.image(image_display, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image using the Flask-inspired function
            img_array = preprocess_image(img_path)

            # Make prediction
            prediction = model.predict(img_array)[0][0]  # Assuming a single-value output like in Flask

            # Use the same threshold logic as Flask
            if prediction > 0.015:
                result = "Non Autistic"
                st.markdown(no_style, unsafe_allow_html=True)
            else:
                result = "Autistic"
                st.markdown(yes_style, unsafe_allow_html=True)

            # Display raw prediction value for debugging/info
            st.write(f"Prediction value: {prediction}")

            # Clean up the temporary file
            os.remove(img_path)

        except Exception as e:
            # st.error("Failed to process the image. Please ensure the image is valid and in a supported format (.jpg, .png, .jpeg).")
            pass

if __name__ == "__main__":
    image_analysis_ui()