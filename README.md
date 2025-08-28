🛡️ Hate Speech Detection on Twitter






📌 Overview

This project is a machine learning pipeline for classifying tweets into:

🟥 Hate Speech

🟧 Offensive Language

🟩 Non-Offensive

It applies NLP preprocessing, feature extraction, and supervised learning to tackle harmful content detection on social media.

📂 Dataset

Source: twitter.csv

Columns:

tweet → raw text from Twitter

class → numeric label (0,1,2)

labels → mapped text labels (Hate Speech, Offensive, No Hate)

⚙️ Workflow

Exploratory Data Analysis (EDA)

Class distribution

Tweet length distribution

Word clouds

Text Preprocessing

Lowercasing, removing URLs, punctuation, numbers

Stopword removal (NLTK)

Stemming (Snowball Stemmer)

Feature Engineering

Bag-of-Words with CountVectorizer

Train/test split

Model Training

Base model: Decision Tree Classifier

Evaluation

Accuracy on train & test sets

Extendable with precision, recall, F1-score

Prediction Demo

Example: "you are a bad person" → Offensive Language

📊 Results

Decision Tree achieved solid accuracy on training/test sets

Train Accuracy: 99.96%

Test Accuracy: 87.68%

The Decision Tree Classifier fits training data almost perfectly, but shows lower test accuracy — indicating possible overfitting.

Despite this, the model demonstrates strong capability to classify tweets into Hate Speech, Offensive, or Non-Offensive categories.

Future improvements (see below) can boost performance significantly

🚀 Future Improvements

Replace BoW with TF-IDF or word embeddings

Experiment with Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost

Handle class imbalance with resampling or class weights

Use Transformer models (BERT/DistilBERT)

Deploy via Streamlit/Flask for real-time moderation

🖥️ How to Run
# Clone repo
git clone https://github.com/divyanshmathur004/hate-speech-detection.git
cd hate-speech-detection

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Hate_Speech_Detection_Model.ipynb
