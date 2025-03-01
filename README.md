# ASD Detection App

This application is designed to assist in the detection of Autism Spectrum Disorder (ASD) through various interactive features, including quizzes, games, and chatbot interactions.

## Directory Structure

- **app/**: Contains the main application files.
  - **__init__.py**: Initialization file for the app package.
  - **main.py**: Main Streamlit app entry point.
  - **chatbot.py**: Handles PDF upload and chatbot interaction.
  - **quiz.py**: Random quiz questions and ML model.
  - **games.py**: Finger tapping, memory, emoji test logic.
  - **pdf_generator.py**: Generates user test reports as PDFs.
  - **utils.py**: Helper functions (loading datasets, etc.).
  - **models/**: Folder to store ML models.
  - **data/**: Folder for datasets.
  - **assets/**: Store any images/icons.
  - **requirements.txt**: Python dependencies.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To run the application, execute:

```
streamlit run app/main.py
