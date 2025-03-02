# app/chatbot.py
import streamlit as st
import fitz  # PyMuPDF for PDF processing
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Handle chat input and generate response
def handle_chat_input(prompt):
    if prompt:
        st.session_state.chat_history.append(("User", prompt)) # Add user message to history
        
        # Generate bot response
        context = f"Chat History: {st.session_state.chat_history}\nUser: {prompt}"
        if st.session_state.get("pdf_text"):
            context += f"\nPDF Context: {st.session_state.pdf_text}"
        context += "\nIf this given pdf context or user latest message is not related to Autism then indicate that this is not related to Autism with a professional response and say that ask anything to related autism spectrum disorder."
        response = model.generate_content(context)
        bot_response = response.text if response else "Sorry, I couldn't process that."
        st.session_state.chat_history.append(("Bot", bot_response))

# Process PDF upload and generate analysis
def process_pdf_upload():
    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is not None and st.session_state.get("pdf_text") is None:
        st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state.analyzing = True  # Set analyzing flag
        st.session_state.chat_history.append(("Bot", "PDF successfully uploaded, analyzing report..."))
        # Generate analysis from Gemini
        context = f"Here is a patient's autism test report:\n{st.session_state.pdf_text}\nPlease provide an analysis and precautions based on this report."
        response = model.generate_content(context)
        bot_response = response.text if response else "I'm unable to analyze the report."
        # Remove analyzing message and add actual response
        st.session_state.chat_history = [msg for msg in st.session_state.chat_history if "analyzing report" not in msg[1]]
        st.session_state.chat_history.append(("Bot", bot_response))
        st.session_state.analyzing = False  # Clear analyzing flag

def chatbot_ui():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False

    st.title("Autism Support Chatbot")

    # Chat container
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role.lower()):
                st.write(message)

    st.markdown(
        """
        <style>
        .stContainer {
            height: 400px; /* Fixed height for scrolling */
            overflow-y: auto;
            padding: 10px;
            border: 2px solid #444; /* Border */
            border-radius: 5px;
            background-color: #f9f9f9; /* Light background */
            margin-bottom: 20px;
        }
        .stChatMessage {
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Chat input at the bottom
    prompt = st.chat_input("Type your message here...")
    if prompt:
        handle_chat_input(prompt)
        st.rerun()

    # PDF upload section
    st.subheader("Upload Your ASD Report")
    st.file_uploader("", type="pdf", key="pdf_uploader", on_change=process_pdf_upload)

    # Rerun if analyzing to update the UI
    if st.session_state.analyzing:
        st.rerun()

if __name__ == "__main__":
    chatbot_ui()