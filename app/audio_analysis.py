import os
import numpy as np
import librosa
import joblib
import streamlit as st
from pydub import AudioSegment
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Define folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

# Create models folder if it doesn't exist
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Define available models and their accuracies
models = {
    'rf.pkl': 'Random Forest', #(~90% accuracy)
    'ann.pkl': 'Artificial Neural Network', #(~72% accuracy)',
    'svm.pkl': 'Support Vector Machine', #(~84% accuracy)
    'nb.pkl': 'Naive Bayes', # (~81% accuracy)
}

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_data, sample_rate, n_mfcc=20):
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc_features

# Streamlit UI for audio analysis
def audio_analysis_ui():
    st.title("Audio Analysis for Autism Detection")

    # Model selection
    model_name = st.selectbox("Choose a model", list(models.values()))
    chosen_model = [k for k, v in models.items() if v == model_name][0]
    model_path = os.path.join(MODELS_FOLDER, chosen_model)

    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file {chosen_model} not found in {MODELS_FOLDER}! Please ensure the .pkl files are in the models folder.")
        return

    # Load the selected model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model {chosen_model}: {str(e)}")
        return

    # File uploader for audio
    uploaded_file = st.file_uploader("Upload an audio file (.m4a)", type="m4a")

    # CSS styles for prediction output
    yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
    no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'

    # Process the uploaded file
    if uploaded_file:
        try:
            # Load and process the audio file
            audio = AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()), format='m4a')
            samples = audio.get_array_of_samples()
            y = np.array(samples).astype(np.float32) / (2**15 - 1)  # Normalize to [-1, 1]
            sr = audio.frame_rate

            # Extract MFCC features
            mfcc_features = extract_mfcc(y, sr)

            # Check for valid MFCC features
            if not np.isnan(mfcc_features).any():
                # Calculate row-wise average of MFCC features
                mfcc_avg = np.mean(mfcc_features, axis=1, keepdims=True)
                mfcc_features_reshaped = mfcc_avg.reshape(1, -1)  # Reshape for prediction

                # Make prediction
                predicted_label = model.predict(mfcc_features_reshaped)
                if predicted_label[0] == 1:
                    st.markdown(yes_style, unsafe_allow_html=True)
                else:
                    st.markdown(no_style, unsafe_allow_html=True)
            else:
                st.error("Could not extract valid MFCC features from the audio file. The file may be corrupted or in an unsupported format.")
        except Exception as e:
            st.error("Failed to process the audio file. It may be corrupted or in an unsupported format. Please try a different .m4a file.")
            # Uncomment the line below if you need the full traceback for debugging
            # st.error(f"Full traceback: {str(e)}")

if __name__ == "__main__":
    audio_analysis_ui()