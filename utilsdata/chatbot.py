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
    """Extracts text from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Handle send action (for both Enter key and Send button)
def handle_send():
    user_input = st.session_state.get("chat_input", "")
    if user_input:
        st.session_state.chat_history.append(("User", user_input))
        
        # Generate bot response
        context = f"Chat History: {st.session_state.chat_history}\nUser: {user_input}"
        if st.session_state.get("pdf_text"):
            context += f"\nPDF Context: {st.session_state.pdf_text}"
        response = model.generate_content(context)
        bot_response = response.text if response else "Sorry, I couldn't process that."
        st.session_state.chat_history.append(("Bot", bot_response))
        
        # Clear input and return to refresh
        st.session_state.chat_input = ""
        return True  # Indicate a refresh is needed

def process_pdf_upload():
    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is not None and st.session_state.get("pdf_text") is None:  # Process only if not already processed
        st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state.pdf_processed = True  # Flag to trigger response
        st.success("PDF uploaded successfully! Analyzing report...")

def chatbot_ui():
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "mode" not in st.session_state:
        st.session_state.mode = "initial"
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    st.title("Autism Support Chatbot")

    # Chat container with scrolling at the top
    st.subheader("Conversation")
    chat_container = st.container()
    with chat_container:
        # Generate initial response if PDF is processed but not yet responded to
        if st.session_state.pdf_processed and not st.session_state.chat_history:
            if st.session_state.pdf_text:
                context = f"Here is a patient's autism test report:\n{st.session_state.pdf_text}\nWhat recommendations do you suggest?"
                response = model.generate_content(context)
                bot_response = response.text if response else "I'm unable to analyze the report."
                st.session_state.chat_history.append(("Bot", bot_response))
                st.session_state.mode = "chat"
                st.session_state.pdf_processed = False  # Reset flag after processing

        for role, message in st.session_state.chat_history:
            if role == "Bot":
                st.markdown(
                    f"""
                    <div style='text-align: left; background-color: #2E2E2E; color: white; padding: 10px; border-radius: 10px; margin: 5px 0; max-width: 70%; display: inline-block;'>
                        {message}
                    """,
                    # </div>
                    # <br clear='both'>
                    unsafe_allow_html=True,
                )
            else:  # User
                st.markdown(
                    f"""
                    <div style='text-align: right; background-color: #4A4A4A; color: white; padding: 10px; border-radius: 10px; margin: 5px 0; max-width: 70%; display: inline-block; float: right;'>
                        {message}
                    </div>
                    <br clear='both'>
                    """,
                    unsafe_allow_html=True,
                )

    # Add CSS for scrolling, fixed height, border, and background
    st.markdown(
        """
        <style>
        .stContainer {
            height: 300px; /* Fixed height for scrolling */
            overflow-y: auto;
            padding: 10px;
            border: 2px solid #444; /* Border */
            border-radius: 5px;
            background-color: #333333; /* Light dark background */
        }
        .stApp {
            max-height: 600px; /* Limit overall page height */
            overflow: hidden;
        }
        </style>
        """,
        # .stTextInput > div > div > input {
        #     height: 60px; /* Increased height for input field */
        #     font-size: 16px;
        # }
        unsafe_allow_html=True,
    )

    # Input field below conversation
    st.subheader("Type Your Messages")
    user_input = st.text_input("", key="chat_input", on_change=handle_send, args=(), help="Press Enter or click Send to submit")

    # Upload section at the bottom
    st.subheader("Upload Your ASD Report")
    st.file_uploader("", type="pdf", key="pdf_uploader", on_change=process_pdf_upload)

    # Send button
    if st.button("Send"):
        if handle_send():
            st.rerun()

if __name__ == "__main__":
    chatbot_ui()