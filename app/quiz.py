import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Load questions from JSON in data folder
def load_questions():
    project_root = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(project_root, "data", "asd-questions.json")
    try:
        with open(questions_path, "r") as f:
            questions_data = json.load(f)
        return questions_data["questions"]
    except FileNotFoundError:
        st.error(f"Questions file (asd-questions.json) not found at {questions_path}.")
        return []

# Load dataset from CSV in data folder
def load_dataset():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "data", "dataset-asd.csv")
    try:
        df = pd.read_csv(dataset_path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file (dataset-asd.csv) not found at {dataset_path}.")
        return None

def train_xgboost_model(df):
    """Train XGBoost model on the dataset."""
    if df is None or df.empty:
        st.error("No dataset available to train the model.")
        return None

    # Prepare features and target
    features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score', 
                'Sex', 'Jaundice', 'Family_mem_with_ASD', 'Ethnicity']
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
    df['Jaundice'] = df['Jaundice'].map({'yes': 1, 'no': 0})
    df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].map({'yes': 1, 'no': 0})
    # Encode Ethnicity as a categorical variable (using one-hot encoding might be better, but we'll use numeric mapping for simplicity)
    ethnicity_map = {
        'asian': 0, 'black': 1, 'hispanic': 2, 'latino': 3, 'middle eastern': 4, 'mixed': 5, 
        'native indian': 6, 'pacifica': 7, 'south asian': 8, 'white european': 9, 'unknown': 10
    }
    df['Ethnicity'] = df['Ethnicity'].str.lower().map(ethnicity_map).fillna(10)  # Default to 'unknown' if not found
    
    X = df[features]
    y = df['Class/ASD Traits '].map({'Yes': 1, 'No': 0})
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

def predict_autism(user_answers, xgb_model, df):
    """Predict autism based on user answers."""
    if xgb_model is None or df is None:
        st.error("Model or dataset not available for prediction.")
        return None, None

    # Map answers to numerical values
    answers = {f'A{i+1}': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(ans) 
               for i, ans in enumerate(user_answers[:9])}
    answers['A10'] = ["Clear and early", "Delayed but clear", "Delayed and unclear", "Still not speaking"].index(user_answers[9])
    answers['Age_Mons'] = int(user_answers[10]) * 12 if user_answers[10].isdigit() else 0  # Convert years to months
    answers['Qchat-10-Score'] = sum(answers[f'A{i+1}'] for i in range(9)) + answers['A10']
    answers['Sex'] = 1 if user_answers[11].lower() == "male" else 0  # 'm': 1, 'f': 0
    answers['Jaundice'] = 1 if user_answers[12].lower() == "yes" else 0
    answers['Family_mem_with_ASD'] = 1 if user_answers[13].lower() == "yes" else 0
    # Map Ethnicity
    ethnicity_map = {
        'asian': 0, 'black': 1, 'hispanic': 2, 'latino': 3, 'middle eastern': 4, 'mixed': 5, 
        'native indian': 6, 'pacifica': 7, 'south asian': 8, 'white european': 9, 'unknown': 10
    }
    answers['Ethnicity'] = ethnicity_map.get(user_answers[14].lower(), 10)  # Default to 'unknown'
    
    # Prepare input for prediction
    input_data = pd.DataFrame([answers])
    
    # Predict
    prediction = xgb_model.predict(input_data)[0]
    probability = xgb_model.predict_proba(input_data)[0][1]  # Probability of autism
    
    return prediction, probability

def save_to_dataset(user_answers, prediction):
    """Save user answers and prediction to dataset-asd.csv."""
    df = load_dataset()
    if df is None:
        return

    # Get the next Case_No
    new_case_no = df['Case_No'].max() + 1 if not df.empty else 1

    # Prepare new row
    new_row = {
        'Case_No': new_case_no,
        'A1': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[0]),
        'A2': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[1]),
        'A3': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[2]),
        'A4': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[3]),
        'A5': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[4]),
        'A6': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[5]),
        'A7': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[6]),
        'A8': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[7]),
        'A9': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[8]),
        'A10': ["Clear and early", "Delayed but clear", "Delayed and unclear", "Still not speaking"].index(user_answers[9]),
        'Age_Mons': int(user_answers[10]) * 12 if user_answers[10].isdigit() else 0,  # Convert years to months
        'Qchat-10-Score': sum(["Always", "Usually", "Sometimes", "Rarely", "Never"].index(ans) for ans in user_answers[:9]) + ["Clear and early", "Delayed but clear", "Delayed and unclear", "Still not speaking"].index(user_answers[9]),
        'Sex': 'm' if user_answers[11].lower() == "male" else 'f',
        'Ethnicity': user_answers[14].lower(),  # Save as string
        'Jaundice': 'yes' if user_answers[12].lower() == "yes" else 'no',
        'Family_mem_with_ASD': 'yes' if user_answers[13].lower() == "yes" else 'no',
        'Who completed the test': user_answers[15].lower(),  # Updated to dropdown value
        'Class/ASD Traits ': 'Yes' if prediction == 1 else 'No'
    }
    
    # Append new row to dataset
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset-asd.csv"), index=False)

def generate_pdf_report(user_answers, prediction, probability, questions):
    """Generate a beautifully formatted PDF report using Gemini API and ReportLab."""
    # Define additional questions explicitly to match the user_answers list
    additional_questions = [
        "What is the age of your child (in years)?",
        "What is the gender of your child?",
        "Was your child born with jaundice?",
        "Does any immediate family member have a history with ASD?",
        "What is the ethnicity of your child?",
        "Who is completing the test?"
    ]
    # Combine the loaded questions (from asd-questions.json) with additional questions
    all_questions = [q["question"] for q in questions] + additional_questions

    # Generate report content using Gemini API
    context = f"""
    Autism Screening Report:
    User Answers: {list(zip(all_questions, user_answers))}
    Prediction: {'Autism' if prediction == 1 else 'No Autism'}
    Probability of Autism: {probability:.2f}
    """
    response = model.generate_content(context)
    report_text = response.text if response else "Unable to generate report content."

    # Parse the response to structure it properly
    if "Autism Screening Report" in report_text:
        report_content = report_text.split("Autism Screening Report:")[1].strip()
    else:
        report_content = report_text

    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    # Customize styles
    styles['Title'].fontSize = 20
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Heading1'].fontSize = 16
    styles['Heading1'].fontName = 'Helvetica-Bold'
    styles['Normal'].fontSize = 12
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].leading = 14

    elements = []

    # Add title
    elements.append(Paragraph("Autism Screening Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Add prediction
    prediction_text = f"<font size=18><b>{'Autism' if prediction == 1 else 'No Autism'}</b></font>"
    elements.append(Paragraph(prediction_text, styles['Heading1']))
    elements.append(Spacer(1, 12))

    # Add probability
    probability_text = f"Probability of Autism: <b>{probability:.2f}</b>"
    elements.append(Paragraph(probability_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Structured content
    elements.append(Paragraph("Report Details", styles['Heading1']))
    elements.append(Spacer(1, 6))

    content_lines = report_content.split("\n")
    for line in content_lines:
        if line.strip():
            if "screening tools are not diagnostic" in line.lower():
                line = f"<b>Screening Tools Are Not Diagnostic:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "false positives are possible" in line.lower():
                line = f"<b>False Positives Are Possible:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "context is missing" in line.lower():
                line = f"<b>Context Is Missing:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "age plays a role" in line.lower():
                line = f"<b>Age Plays a Role:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "next steps" in line.lower():
                line = f"<b>Next Steps:</b> {line.split(':', 1)[1] if ':' in line else line}"

            elements.append(Paragraph(line, styles['Normal']))
            elements.append(Spacer(1, 6))

    # Add User Answers with Questions
    elements.append(Paragraph("User Answers:", styles['Heading1']))
    elements.append(Spacer(1, 6))
    for i, (question, answer) in enumerate(zip(all_questions, user_answers), 1):
        # Format as "Q#: [Question] - [Answer]" with answer in bold
        answer_text = f"Q{i}: {question} - <b>{answer}</b>"
        elements.append(Paragraph(answer_text, styles['Normal']))
        elements.append(Spacer(1, 4))

    doc.build(elements)
    buffer.seek(0)
    return buffer
def quiz_ui():
    st.title("Autism Quiz Test")

    # Load questions and dataset
    questions = load_questions()
    df = load_dataset()
    if not questions or df is None:
        return

    # Train XGBoost model
    xgb_model = train_xgboost_model(df)

    # Collect user responses for the 10 questions
    user_answers = []
    st.subheader("Please answer the following questions:")
    for q in questions:
        answer = st.selectbox(q["question"], q["options"], key=f"q{q['id']}")
        user_answers.append(answer)

    # Additional questions
    age = st.text_input("What is the age of your child (in years)?", key="age")
    gender = st.selectbox("What is the gender of your child?", ["Male", "Female"], key="gender")
    jaundice = st.selectbox("Was your child born with jaundice?", ["Yes", "No"], key="jaundice")
    family_history = st.selectbox("Does any immediate family member have a history with ASD?", ["Yes", "No"], key="family_history")
    ethnicity = st.selectbox("What is the ethnicity of your child?", 
                             ["Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Mixed", 
                              "Native Indian", "Pacifica", "South Asian", "White European", "Unknown"], 
                             key="ethnicity")
    completed_by = st.selectbox("Who is completing the test?", 
                                ["Family Member", "Self", "Mother", "Health Care Professional", "Others"], 
                                key="completed_by")

    # Add to user_answers for prediction
    user_answers.extend([age, gender, jaundice, family_history, ethnicity, completed_by])

    # Submit button
    if st.button("Submit Quiz"):
        if all(len(ans) > 0 for ans in user_answers[:9]) and age:
            prediction, probability = predict_autism(user_answers, xgb_model, df)
            if prediction is not None:
                st.session_state.prediction = prediction
                st.session_state.probability = probability

                # Display result
                if prediction == 0:
                    st.success(f"Result: No Autism (Probability of Autism: {probability:.2f})", icon="✅")
                else:
                    st.error(f"Result: Autism (Probability of Autism: {probability:.2f})", icon="⚠️")
                
                # Save to dataset
                save_to_dataset(user_answers, prediction)
        else:
            st.error("Please answer all questions before submitting.")

    # Generate Report button
    if st.button("Generate Report"):
        if 'prediction' in st.session_state and 'probability' in st.session_state:
            pdf_buffer = generate_pdf_report(user_answers, st.session_state.prediction, st.session_state.probability, questions)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="autism_screening_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Please submit the quiz first to generate a report.")

if __name__ == "__main__":
    quiz_ui()