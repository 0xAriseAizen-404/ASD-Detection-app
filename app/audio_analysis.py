import os
import numpy as np
import librosa
import joblib
import streamlit as st
from pydub import AudioSegment
import io
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

models = {
    'rf.pkl': 'Random Forest',
    'ann.pkl': 'Artificial Neural Network',
    'svm.pkl': 'Support Vector Machine',
    'nb.pkl': 'Naive Bayes',
}

def extract_mfcc(audio_data, sample_rate, n_mfcc=20):
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc_features

def audio_analysis_ui():
    st.title("Audio Analysis for Autism Detection")
    model_name = st.selectbox("Choose a model", list(models.values()))
    chosen_model = [k for k, v in models.items() if v == model_name][0]
    model_path = os.path.join(MODELS_FOLDER, chosen_model)
    if not os.path.exists(model_path):
        st.error(f"Model file {chosen_model} not found in {MODELS_FOLDER}! Please ensure the .pkl files are in the models folder.")
        return
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model {chosen_model}: {str(e)}")
        return
    uploaded_file = st.file_uploader("Upload an audio file (.m4a)", type="m4a")
    yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
    no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'
    if uploaded_file:
        try:
            audio = AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()), format='m4a')
            samples = audio.get_array_of_samples()
            y = np.array(samples).astype(np.float32) / (2**15 - 1)
            sr = audio.frame_rate
            mfcc_features = extract_mfcc(y, sr)
            if not np.isnan(mfcc_features).any():
                mfcc_avg = np.mean(mfcc_features, axis=1, keepdims=True)
                mfcc_features_reshaped = mfcc_avg.reshape(1, -1)
                predicted_label = model.predict(mfcc_features_reshaped)
                if predicted_label[0] == 1:
                    st.markdown(yes_style, unsafe_allow_html=True)
                    result = "Autistic"
                else:
                    st.markdown(no_style, unsafe_allow_html=True)
                    result = "Non Autistic"
                # Store audio analysis data in pdf_text
                audio_data = f"Audio Analysis Results:\nFile: {uploaded_file.name}\nModel: {model_name}\nPrediction: {result}\n\n"
                st.session_state.pdf_text += audio_data
            else:
                st.error("Could not extract valid MFCC features from the audio file.")
        except Exception as e:
            st.error("Failed to process the audio file. Please try a different .m4a file.")

if __name__ == "__main__":
    audio_analysis_ui()