import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset (you can replace with Kaggle dataset later)
data = {
    "text": [
        "Government confirms new policy",
        "Aliens landed in India yesterday",
        "Stock market hits new high",
        "Fake miracle cure for cancer discovered",
        "New education reforms announced"
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")
