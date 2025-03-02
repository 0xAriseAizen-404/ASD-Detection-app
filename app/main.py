import streamlit as st
from chatbot import chatbot_ui
from quiz import quiz_ui
from games import games_ui
from audio_analysis import audio_analysis_ui
from image_analysis import image_analysis_ui

st.set_page_config(page_title="ASD Detection App", layout="wide")

with st.sidebar:
    st.title("🔴 ASD Detection App")

    with st.expander("⭐ APPS", expanded=True):
        page = st.radio(
            "Go to",
            ["Chatbot", "Quiz Test", "Gaming Tests", "Audio Analysis", "Image Analysis"],
            label_visibility="collapsed"
        )

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


# import streamlit as st
# from games import games_ui  # Assuming this handles your games section
# from chatbot import chatbot_ui  # Assuming this handles your chatbot section
# from quiz import quiz_ui  # Assuming this handles your quiz section
# from audio_analysis import audio_analysis_ui  # Import the audio analysis UI

# # Set up the sidebar for navigation
# st.sidebar.title("ASD Detection App")
# page = st.sidebar.selectbox("Choose a page", ["Home", "Games", "Chatbot", "Quiz", "Audio Analysis"])

# # Display the selected page
# if page == "Home":
#     st.title("Welcome to the ASD Detection App")
#     st.write("Use the sidebar to navigate to different sections of the app.")
# elif page == "Games":
#     games_ui()
# elif page == "Chatbot":
#     chatbot_ui()
# elif page == "Quiz":
#     quiz_ui()
# elif page == "Audio Analysis":
#     audio_analysis_ui()