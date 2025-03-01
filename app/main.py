import streamlit as st
from chatbot import chatbot_ui
from quiz import quiz_ui
from games import games_ui
from audio_analysis import audio_analysis_ui
from image_analysis import image_analysis_ui

st.set_page_config(page_title="ASD Detection App", layout="wide")

# Sidebar with categorized navigation
with st.sidebar:
    st.title("🔴 ASD Detection App")

    with st.expander("⭐ APPS", expanded=True):
        page = st.radio(
            "Go to",
            ["Chatbot", "Quiz Test", "Gaming Tests", "Audio Analysis", "Image Analysis"],
            label_visibility="collapsed"
        )

    # Placeholder for additional expanders (as in the screenshot)
    # with st.expander("🌟 COMPONENTS", expanded=True):
    #     st.write("No components available.")

if page == "Chatbot":
    chatbot_ui()
elif page == "Quiz Test":
    quiz_ui()
elif page == "Gaming Tests":
    games_ui()
elif page == "Audio Analysis":
    audio_analysis_ui()
elif page == "Image Analysis":
    image_analysis_ui()