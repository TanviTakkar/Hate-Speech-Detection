# Hate Speech Detection Model and Streamlit Application

## Project Overview

This project aims to build a robust machine learning model to detect hate speech in text and deploy it as an accessible web application. The objective is to provide a tool that can be used to identify and flag harmful content, helping to create a safer online environment.

## Project Aim

The primary goal of this project is to develop a system that can accurately classify text as either hateful or non-hateful. The project leverages state-of-the-art natural language processing techniques, specifically fine-tuning a pre-trained DistilBERT model, to achieve high accuracy in hate speech detection. By deploying the model through a user-friendly Streamlit application, we aim to make this tool easily accessible for various applications, including content moderation, research, and public awareness.

## What We Did

1. **Data Collection and Preparation:**
    - We used a balanced dataset containing text samples labeled as either hateful or non-hateful. The dataset was pre-processed to remove noise, such as HTML tags, special characters, and stopwords.
    - The dataset was obtained from [Kaggle: Hate Speech Detection Curated Dataset](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset/data?select=HateSpeechDatasetBalanced.csv).
    - We also performed data augmentation and sampling to ensure the model was trained on a diverse and representative dataset.

2. **Model Development:**
    - We fine-tuned a DistilBERT model, a lighter version of BERT, specifically designed for text classification tasks.
    - The model was trained on the pre-processed dataset and optimized using techniques such as learning rate scheduling and weight decay to improve performance.

3. **API and Web App Development:**
    - A Flask-based API was developed to serve the model predictions. This API can be accessed programmatically to classify text as hateful or non-hateful.
    - A Streamlit application was created to provide a graphical interface where users can input text and receive real-time predictions. This makes the tool accessible even to non-technical users.

4. **Model Interpretability:**
    - We integrated LIME (Local Interpretable Model-agnostic Explanations) to help explain the predictions made by the model. This feature provides insights into which words or phrases in the input text contributed most to the model’s decision, making the model’s behavior more transparent.

## How We Did It

## 1. **Data Preprocessing**
    - ## **Cleaning:** We removed unnecessary HTML tags, special characters, and digits from the text data.
    - ## **Tokenization and Stopword Removal:** We tokenized the text and removed common English stopwords that do not contribute meaningfully to the classification task.
    - ## **Spelling Correction:** We used the SymSpell library to correct any spelling errors in the text.

## 2. **Model Training**
    - **Model Selection:** We selected DistilBERT due to its balance between performance and computational efficiency.
    - **Fine-tuning:** We fine-tuned the DistilBERT model on our dataset, adjusting the model’s parameters to minimize classification error.
    - **Evaluation:** The model’s performance was evaluated using accuracy as the primary metric, with regular evaluations after each training epoch.

## 3. **API Development**
    - **Flask API:** The model was integrated into a Flask API, which allows users to send HTTP POST requests with text data and receive predictions in response.
    - **End-to-End Pipeline:** The API handles the entire process from text preprocessing to model prediction and returns the classification result.

## 4. **Streamlit Application**
    - **User Interface:** We built a Streamlit app to provide an intuitive interface for text classification. Users can enter text directly into the app and view the prediction results instantly.
    

## 5. **Deployment**
    - The Streamlit app and API were deployed on a local server, making the model accessible for testing and demonstration purposes. Future plans include deploying the application to a cloud service for broader accessibility.

## Installation

### Prerequisites

- Python 3.7 or higher
- Streamlit
- Additional Python libraries: `pandas`, `matplotlib`, `seaborn`, `transformers`, `torch`, `lime`, `bs4`

### Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/hate-speech-detection.git
    cd hate-speech-detection
    ```

2. **Install Required Libraries**

    Manually install the required Python libraries:

    ```bash
    pip install pandas matplotlib seaborn transformers torch lime bs4
    ```

3. **Run the Streamlit App**

    ```bash
    streamlit run hate_speech_api.py
    ```

    This will launch the Streamlit application in your browser.

## Usage

### Streamlit App

- Input any text into the provided textbox to get a prediction on whether the text is hateful or non-hateful.

### API Usage

The API can be used programmatically to classify text. Example request:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"text": "Your input text here"}' \
    http://localhost:8501/predict
