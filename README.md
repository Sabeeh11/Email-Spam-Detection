# Email Spam Detection using NLP and Machine Learning

## Overview
This project classifies messages as spam or not spam using Natural Language Processing and machine learning.

## Dataset

Download from:
https://www.kaggle.com/datasets/venky73/spam-mails-dataset

Place it inside:
data/spam_ham_dataset.csv

## Tech Stack
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit

## Model Pipeline
1. Load dataset
2. Clean labels
3. Convert text to TF-IDF features
4. Train Logistic Regression model
5. Evaluate model
6. Deploy with Streamlit

## Results

| Metric | Score |
|--------|------|
| Accuracy | 0.9845 |
| Precision (Spam) | 0.97 |
| Recall (Spam) | 0.98 |
| F1-score (Spam) | 0.97 |

## How to Run

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
streamlit run app.py

## Live Demo

Try the app here:
https://your-link.streamlit.app](https://email-spam-detection-gwd2hpptmtm3gyg6quesrt.streamlit.app/
