
# Hate Speech Analysis Project

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Preprocessing](#data-preprocessing)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Topic Modeling](#topic-modeling)
9. [Model Training](#model-training)
10. [Model Evaluation](#model-evaluation)
11. [Interpretation of Results](#interpretation-of-results)
12. [License](#license)

## Introduction

The Hate Speech Analysis Project aims to classify text content as hateful or non-hateful while extracting topics related to hate speech. This project utilizes state-of-the-art natural language processing (NLP) techniques and machine learning models to effectively analyze and interpret text data.

## Features

- **Text Classification**: Classifies content into hateful and non-hateful categories.
- **Topic Extraction**: Identifies and summarizes topics related to hate speech using BERTopic.
- **Data Preprocessing**: Cleans and prepares the text data for analysis.
- **Exploratory Data Analysis**: Visualizes data distributions and identifies common words in the dataset.
- **Model Interpretation**: Provides insights into model predictions using LIME (Local Interpretable Model-agnostic Explanations).

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - Matplotlib
  - Seaborn
  - NLTK
  - Transformers (Hugging Face)
  - BERTopic
  - LIME
  - Scikit-learn
  - SymSpell

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hate-speech-analysis.git
   cd hate-speech-analysis
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project, execute the following command:
```bash
python main.py
```

## Data Preprocessing

The data preprocessing steps include:
- Removing special characters and stop words
- Tokenization
- Lemmatization

## Exploratory Data Analysis (EDA)

This section covers visualizations that depict the distribution of hate speech in the dataset, common words, and other relevant insights.

## Topic Modeling

Utilized BERTopic for extracting topics related to hate speech.

## Model Training

Trained the model using DistilBERT and TopicBERT, fine-tuning them on the dataset for optimal performance.

## Model Evaluation

Evaluated the model's performance using accuracy, precision, recall, and F1-score metrics.

## Interpretation of Results

Interpreted model predictions using LIME to provide insights into the decision-making process of the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
