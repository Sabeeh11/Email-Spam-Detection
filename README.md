# Email Spam Detection using NLP and Machine Learning

## Overview
This project classifies messages as spam or not spam using Natural Language Processing and machine learning.

## Dataset
The project uses a labeled SMS/email spam dataset with two classes:
- Ham: legitimate message
- Spam: unwanted promotional or malicious message

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
|---|---|
| Accuracy | Add your score |
| Precision | Add your score |
| Recall | Add your score |
| F1-score | Add your score |

## How to Run

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
streamlit run app.py
