import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()
    text = text.strip()
    return text


df = pd.read_csv("spam_ham_dataset.csv")

df = df.drop(columns=["Unnamed: 0", "label_num"], errors="ignore")

df = df[["label", "text"]]
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["clean_message"] = df["message"].apply(clean_text)

X = df["clean_message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

joblib.dump(model, "spam_model.pkl")

print("Model trained and saved as spam_model.pkl")
