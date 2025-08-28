🛡️ Hate Speech Detection Model
📌 Project Overview

This project focuses on building a machine learning model to detect hate speech, offensive language, and non-offensive tweets from Twitter data.
The motivation is to tackle the growing issue of online harassment and provide an automated way to classify harmful content.

📂 Dataset

Source: [twitter.csv] (provided dataset)

Classes:

0 → Hate Speech

1 → Offensive Language

2 → No Hate and Offensive

Each tweet is labeled accordingly to train the model.

⚙️ Project Workflow

Data Loading → Import CSV file using Pandas.

EDA (Exploratory Data Analysis)

Class distribution visualization

Tweet length analysis

Word clouds of common words in each class

Text Preprocessing

Lowercasing text

Removing punctuation, URLs, HTML tags, numbers

Removing stopwords (using NLTK)

Applying stemming (Snowball Stemmer)

Feature Engineering

Bag-of-Words representation using CountVectorizer

Optionally extendable to TF-IDF or n-grams

Model Training

Trained using Decision Tree Classifier

Dataset split: 67% training / 33% testing

Evaluation

Train and Test accuracy scores

Can be extended with precision, recall, F1-score, and confusion matrix

Prediction Example

Test the model on a custom input tweet

📊 Results

Decision Tree Model trained and tested with promising accuracy.

Shows ability to correctly differentiate between hate speech, offensive language, and normal tweets.

Future improvements can significantly boost performance.

🚀 Future Improvements

Use TF-IDF Vectorizer instead of simple Bag-of-Words

Train multiple models: Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost

Handle class imbalance with resampling or class weighting

Use advanced NLP methods like Word2Vec, GloVe, or Transformer-based models (BERT)

Deploy as a Flask/Streamlit web app for real-time predictions

🖥️ How to Run

Clone this repository:

git clone https://github.com/your-username/hate-speech-detection.git
cd hate-speech-detection


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook Hate_Speech_Detection_Model.ipynb


(Optional) Test with a custom input:

sample = "you are a bad person"
data = cv.transform([sample]).toarray()
print(model.predict(data))
