Below is a comprehensive `README.md` file in Markdown format that documents everything we’ve done for our `ASD-Detection-App` project. It includes the folder structure, descriptions of all features (chatbot, quiz, games, audio analysis, image analysis), datasets, models, algorithms, installation instructions, package dependencies, and how to run the app.

---

# ASD-Detection-App

## Overview

The **ASD-Detection-App** is a web-based application built using Streamlit to assist in the detection of Autism Spectrum Disorder (ASD) through various interactive modules. The app includes the following features:

- **Chatbot:** A conversational AI chatbot for user interaction.
- **Quiz Test:** A quiz to assess user responses related to ASD traits.
- **Gaming Tests:** Three interactive games (Emoji Game, Memory Game, Ball Clicker Game) to evaluate user behavior.
- **Audio Analysis:** Uses audio data to predict ASD traits with four pre-trained machine learning models (Random Forest, SVM, Naive Bayes, ANN).
- **Image Analysis:** Uses image data to predict ASD traits with a pre-trained deep learning model (ResNet50).

The app is designed to provide a user-friendly interface for parents, caregivers, or professionals to explore potential ASD indicators through multiple modalities.

---

## Folder Structure

The project is organized as follows:

```
ASD-Detection-App/
├── .vscode/                    # VSCode settings (if any)
├── app/                        # Main application directory
│   ├── __pycache__/            # Python cache files
│   ├── assets/                 # Static assets for games
│   │   ├── ball_clicker_game.html
│   │   ├── emoji_game.html
│   │   ├── memory_game.html
│   ├── data/                   # Dataset directory
│   │   ├── train/              # Training images
│   │   │   ├── Autistic/
│   │   │   ├── Non_Autistic/
│   │   ├── test/               # Test images
│   │   │   ├── Autistic/
│   │   │   ├── Non_Autistic/
│   │   ├── valid/              # Validation images
│   │   │   ├── Autistic/
│   │   │   ├── Non_Autistic/
│   ├── features/               # Extracted audio features (MFCC) as .npy files
│   ├── models/                 # Pre-trained models
│   │   ├── rf.pkl              # Random Forest model for audio analysis
│   │   ├── svm.pkl             # Support Vector Machine model for audio analysis
│   │   ├── nb.pkl              # Naive Bayes model for audio analysis
│   │   ├── ann.pkl             # Artificial Neural Network model for audio analysis
│   │   ├── resnet50.h5         # ResNet50 model for image analysis
│   │   ├── vgg16.h5            # VGG16 model for image analysis
│   │   ├── z.h5                # Another model for image analysis
│   ├── notebooks/              # Jupyter notebooks
│   ├── recordings/             # Raw audio files (.m4a)
│   │   ├── aut_Recording_*.m4a # Autistic audio samples
│   │   ├── non_*.m4a          # Non-Autistic audio samples
│   ├── static/                 # Static images for emoji game
│   │   ├── happy.png
│   │   ├── scared.png
│   │   ├── shocked.png
│   │   ├── angry.png
│   │   ├── crying.png
│   │   ├── laughing.png
│   │   ├── sleepy.png
│   │   ├── thinking.png
│   ├── .env                    # Environment variables (e.g., API keys)
│   ├── audio_analysis.py       # Audio analysis module
│   ├── chatbot.py              # Chatbot module
│   ├── games.py                # Games module
│   ├── image_analysis.py       # Image analysis module
│   ├── main.py                 # Main Streamlit app entry point
│   ├── quiz.py                 # Quiz module
│   ├── venv/                   # Virtual environment
│   ├── .gitignore              # Git ignore file
│   ├── README.md               # This file
│   ├── requirements.txt        # Python package dependencies
```

---

## Features

### 1. Chatbot

- **File:** `chatbot.py`
- **Description:** A conversational AI chatbot that interacts with users, providing information or answering questions related to ASD.
- **Implementation:** Uses the `google-generativeai` package to integrate with a generative AI model (e.g., Google Gemini). API keys are stored in the `.env` file and loaded using `dotenv`.
- **Functionality:** Users can input text, and the chatbot responds intelligently, offering insights or guidance.

### 2. Quiz Test

- **File:** `quiz.py`
- **Description:** A questionnaire designed to assess potential ASD traits through user responses.
- **Implementation:** Built using Streamlit, presenting a series of questions with multiple-choice answers.
- **Functionality:** Users answer questions, and the app evaluates responses to provide a summary or score indicating potential ASD traits.

### 3. Gaming Tests

- **File:** `games.py`
- **Description:** Three interactive games to evaluate user behavior and responses, which may indicate ASD traits.
- **Games:**
  - **Emoji Game:** Users interact with emoji-based challenges stored in `assets/emoji_game.html`. Uses images from the `static` folder (`happy.png`, `scared.png`, etc.).
  - **Memory Game:** A memory challenge stored in `assets/memory_game.html`.
  - **Ball Clicker Game:** A reaction-based game stored in `assets/ball_clicker_game.html`.
- **Implementation:** The games are HTML-based and embedded in the Streamlit app using `st.components.v1.html`. The `games.py` module orchestrates the display and interaction.

### 4. Audio Analysis

- **File:** `audio_analysis.py`
- **Description:** Analyzes audio files to predict ASD traits using four pre-trained machine learning models.
- **Models:**
  - **Random Forest (rf.pkl):** ~90% accuracy
  - **Support Vector Machine (svm.pkl):** ~54% accuracy
  - **Naive Bayes (nb.pkl):** ~81% accuracy
  - **Artificial Neural Network (ann.pkl):** ~72% accuracy
- **Dataset:** Audio files (`.m4a`) are stored in the `recordings` folder, with filenames indicating `aut_` for Autistic and `non_` for Non-Autistic samples.
- **Implementation:**
  - Uses `pydub` to load `.m4a` files.
  - Uses `librosa` to extract MFCC (Mel-frequency cepstral coefficients) features.
  - Uses `joblib` to load the pre-trained `.pkl` models.
  - Models predict based on MFCC features, outputting "Autistic" or "Non Autistic".
- **Functionality:** Users upload an `.m4a` file, select a model, and the app displays the prediction ("Autistic" in red or "Non Autistic" in green).

### 5. Image Analysis

- **File:** `image_analysis.py`
- **Description:** Analyzes images to predict ASD traits using deep learning models.
- **Models:**
  - **ResNet50 (resnet50.h5):** A pre-trained ResNet50 model fine-tuned on the dataset.
  - **VGG16 (vgg16.h5):** A pre-trained VGG16 model fine-tuned on the dataset.
- **Dataset:** Images are stored in the `data` folder, split into `train`, `test`, and `valid` subfolders, each containing `Autistic` and `Non_Autistic` subfolders with `.jpg`, `.png`, or `.jpeg` images.
- **Implementation:**
  - Uses `tensorflow` to load and predict with the deep learning models.
  - Uses `Pillow` (PIL) to load and preprocess images (resize to 224x224, convert to RGB).
  - Uses `tensorflow.keras.applications.vgg16.preprocess_input` for VGG16 models (or similar for other architectures).
  - Images are preprocessed to match the model’s input requirements (e.g., 224x224x3 for VGG16).
- **Functionality:** Users upload an image, and the app displays the prediction ("Autistic" in red or "Non Autistic" in green) along with prediction probabilities.

---

## Datasets

### Audio Dataset

- **Location:** `app/recordings/`
- **Structure:** Contains `.m4a` audio files with filenames indicating the class:
  - `aut_Recording_*.m4a`: Autistic samples.
  - `non_*.m4a`: Non-Autistic samples.
- **Usage:** Used for audio analysis to predict ASD traits by extracting MFCC features.

### Image Dataset

- **Location:** `app/data/`
- **Structure:**
  - `train/`: Training images.
    - `Autistic/`: Autistic images.
    - `Non_Autistic/`: Non-Autistic images.
  - `test/`: Test images.
    - `Autistic/`: Autistic images.
    - `Non_Autistic/`: Non-Autistic images.
  - `valid/`: Validation images.
    - `Autistic/`: Autistic images.
    - `Non_Autistic/`: Non-Autistic images.
- **Usage:** Used for image analysis to predict ASD traits using deep learning models.

---

## Models

### Audio Analysis Models

- **Location:** `app/models/`
- **Files:**
  - `rf.pkl`: Random Forest model (~90% accuracy).
  - `svm.pkl`: Support Vector Machine model (~54% accuracy).
  - `nb.pkl`: Naive Bayes model (~81% accuracy).
  - `ann.pkl`: Artificial Neural Network model (~72% accuracy).
- **Training:** They were trained on MFCC features extracted from audio data using `scikit-learn`.

### Image Analysis Models

- **Location:** `app/models/`
- **Files:**
  - `resnet50.h5`: ResNet50 model, fine-tuned on the image dataset.
  - `vgg16.h5`: VGG16 model, fine-tuned on the image dataset.
- **Training:** They were fine-tuned on the image dataset using transfer learning with TensorFlow/Keras. ResNet50 and VGG16 were initially pre-trained on ImageNet, then fine-tuned for binary classification (Autistic vs. Non_Autistic).

---

## Algorithms

### Audio Analysis

- **Feature Extraction:** MFCC features are extracted using `librosa`.
- **Models:**
  - **Random Forest:** An ensemble method using decision trees (`scikit-learn`).
  - **Support Vector Machine (SVM):** A kernel-based classifier (`scikit-learn`).
  - **Naive Bayes:** A probabilistic classifier (`scikit-learn`).
  - **Artificial Neural Network (ANN):** A multi-layer perceptron (`scikit-learn`).
- **Prediction:** Models predict "Autistic" or "Non_Autistic" based on MFCC features.

### Image Analysis

- **Feature Extraction:** Images are preprocessed (resized to 224x224, converted to RGB, normalized) using `tensorflow.keras.applications` preprocessing functions.
- **Models:**
  - **ResNet50:** A deep residual network with 50 layers, fine-tuned for binary classification.
  - **VGG16:** A deep convolutional network with 16 layers, fine-tuned for binary classification.
- **Prediction:** Models predict "Autistic" or "Non_Autistic" based on image features, outputting probabilities for each class.

### Other Algorithms

- **Chatbot:** Uses `google-generativeai` for generative AI responses.
- **Quiz:** Simple rule-based scoring logic implemented in Python.
- **Games:** HTML/JavaScript-based games with user interaction logic.

---

## Installation and Setup

### Prerequisites

- **Python:** Version 3.8–3.11 (TensorFlow is compatible with these versions).
- **ffmpeg:** Required for audio processing (`pydub` and `librosa` dependency).

#### Install ffmpeg on Windows

1. Download `ffmpeg` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (`ffmpeg-release-essentials.zip`).
2. Extract to a folder (e.g., `C:\ffmpeg\ffmpeg-7.1-essentials_build`).
3. Add the `bin` folder to your system PATH:
   - Right-click Start > System > Advanced system settings > Environment Variables.
   - Edit `Path` under "System variables" and add `C:\ffmpeg\ffmpeg-7.1-essentials_build\bin`.
   - Click OK and restart your terminal.
4. Verify installation:
   ```
   ffmpeg -version
   ```

### Step 1: Clone the Repository

Clone the repository (if hosted on GitHub) or copy the project folder to your local machine.

### Step 2: Set Up a Virtual Environment

1. Navigate to the project directory:
   ```
   cd ASD-Detection-App/app
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

### Step 3: Install Dependencies

The project requires the following Python packages (listed in `requirements.txt`):

```
numpy
streamlit
pandas
librosa
joblib
tensorflow
scikit-learn
google-generativeai
python-dotenv
pymupdf
reportlab
xgboost
pydub
Pillow
```

Install them by running:

```
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install the packages manually:

```
pip install numpy streamlit pandas librosa joblib tensorflow scikit-learn google-generativeai python-dotenv pymupdf reportlab xgboost pydub Pillow
```

### Step 4: Configure Environment Variables

- Create a `.env` file in the `app` directory.
- Add any required API keys (e.g., for `google-generativeai`):
  ```
  GOOGLE_API_KEY=your_google_api_key_here
  ```
- The app uses `dotenv` to load these variables.

---

## How to Run the App

1. **Activate the Virtual Environment:**

   ```
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Run the Streamlit App:**

   ```
   streamlit run app/main.py
   ```

3. **Access the App:**
   - Open your browser and go to `http://localhost:8501` (or the URL provided by Streamlit).
   - Use the sidebar to navigate between the Chatbot, Quiz Test, Gaming Tests, Audio Analysis, and Image Analysis pages.

---

## Usage

### Chatbot

- Navigate to the "Chatbot" page.
- Enter text to interact with the AI chatbot.

### Quiz Test

- Navigate to the "Quiz Test" page.
- Answer the questions to receive a summary or score.

### Gaming Tests

- Navigate to the "Gaming Tests" page.
- Choose a game (Emoji Game, Memory Game, or Ball Clicker Game) and follow the instructions.

### Audio Analysis

- Navigate to the "Audio Analysis" page.
- Select a model (Random Forest, SVM, Naive Bayes, or ANN).
- Upload an `.m4a` audio file.
- The app will display the prediction ("Autistic" or "Non Autistic").

### Image Analysis

- Navigate to the "Image Analysis" page.
- Upload an image (`.jpg`, `.png`, `.jpeg`).
- The app will display the prediction ("Autistic" or "Non Autistic") using the `z.h5` model (assumed to be VGG16).

---

## Troubleshooting

### Audio Analysis Issues

- **Error: `[WinError 2] The system cannot find the file specified`**
  - Ensure `ffmpeg` is installed and added to your system PATH (see Prerequisites).
- **Error: `Decoding failed. ffmpeg returned error code`**
  - The uploaded `.m4a` file might be corrupted. Test with a different file or convert the file using FFmpeg:
    ```
    ffmpeg -i input.m4a -c:a aac -b:a 192k output.m4a
    ```

### Image Analysis Issues

- **Error: `Failed to process the image`**
  - Ensure the image is valid (open it in an image viewer).
  - Check the file format (should be `.jpg`, `.png`, or `.jpeg`).
- **Incorrect Predictions:** If the model consistently predicts "Autistic" or "Non_Autistic":
  - Check the class index mapping in `image_analysis.py` (adjust `if predicted_label == 0` logic).
  - Verify the preprocessing matches the model’s training (e.g., `vgg16.preprocess_input` for VGG16 models).
  - Retrain the model on your dataset if necessary (see `train_vgg16.py` or `train_resnet50.py` scripts).

### General Issues

- **Package Version Conflicts:** If you encounter dependency errors, pin specific versions in `requirements.txt` (e.g., `tensorflow==2.10.0`).
- **TensorFlow Compatibility:** Ensure your Python (3.8–3.11) and TensorFlow versions are compatible. Use `pip show tensorflow` to check the version.

---

## Future Improvements

- **Add More Models:** Include additional models for audio and image analysis (e.g., XGBoost for audio, Inception for images).
- **Improve Chatbot:** Enhance the chatbot with more advanced conversational capabilities.
- **Expand Quiz:** Add more questions and detailed scoring for the quiz.
- **Visualizations:** Add visualizations for audio MFCC features or image feature maps.
- **User Authentication:** Implement user login to save results or preferences.

---

## Contributors

- **Mahesh Avvaru:** Developer of the ASD-Detection-App.
<!-- - **Grok 3 (xAI):** Assisted in debugging, implementing features. -->

---

## License

This project is licensed under the MIT License. <!-- See the `LICENSE` file for details. -->

---
