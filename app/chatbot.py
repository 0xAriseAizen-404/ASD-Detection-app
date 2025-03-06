import streamlit as st
import fitz
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def handle_chat_input(prompt):
    if prompt:
        st.session_state.chat_history.append(("User", prompt))
        context = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history])
        if st.session_state.pdf_text:
            context += f"\n\nCollected Data:\n{st.session_state.pdf_text}"
        context += "\n\nIf this context or the latest user message is not related to Autism, respond professionally stating that the chatbot is for Autism-related queries."
        response = model.generate_content(context)
        bot_response = response.text if response else "Sorry, I couldn't process that."
        st.session_state.chat_history.append(("Bot", bot_response))

def process_pdf_upload():
    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state.pdf_text = pdf_text
        st.session_state.chat_history.append(("Bot", "PDF successfully uploaded and analyzed."))
        context = f"Extracted Autism test report:\n{pdf_text}\n\nPlease provide an analysis and precautions based on this report."
        response = model.generate_content(context)
        bot_response = response.text if response else "I'm unable to analyze the report."
        st.session_state.chat_history.append(("Bot", bot_response))

def generate_comprehensive_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Comprehensive ASD Screening Report", styles['Title']),
        Spacer(1, 12),
        Paragraph("Collected Data Analysis", styles['Heading1']),
        Spacer(1, 6)
    ]
    if st.session_state.pdf_text:
        context = f"Collected Data for Analysis:\n{st.session_state.pdf_text}\n\nProvide a detailed analysis and recommendations."
        response = model.generate_content(context)
        analysis = response.text if response else "Unable to generate analysis."
        elements.append(Paragraph("Analysis:", styles['Heading1']))
        elements.append(Spacer(1, 6))
        for line in analysis.split("\n"):
            if line.strip():
                elements.append(Paragraph(line, styles['Normal']))
                elements.append(Spacer(1, 4))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Raw Data:", styles['Heading1']))
        elements.append(Spacer(1, 6))
        for line in st.session_state.pdf_text.split("\n"):
            if line.strip():
                elements.append(Paragraph(line, styles['Normal']))
                elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("No data collected yet from previous pages.", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def chatbot_ui():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    
    st.title("Autism Support Chatbot")
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role.lower()):
                st.write(message)
    
    st.subheader("Upload Your ASD Report")
    st.file_uploader("", type="pdf", key="pdf_uploader", on_change=process_pdf_upload)
    
    prompt = st.chat_input("Type your message here...")
    if prompt:
        handle_chat_input(prompt)
        st.rerun()
    
    if st.button("Generate Comprehensive Report"):
        pdf_buffer = generate_comprehensive_pdf_report()
        st.download_button(
            label="Download Comprehensive Report",
            data=pdf_buffer,
            file_name="comprehensive_asd_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    chatbot_ui()