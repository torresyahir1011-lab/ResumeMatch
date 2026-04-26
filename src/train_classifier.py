import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from load_data import load_data

def train():
    resumes, _ = load_data()

    X = resumes['text']
    y = resumes['label']

    # ----------------------
    # Train/test split
    # ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------
    # Vectorize text
    # ----------------------
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ----------------------
    # Train model
    # ----------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # ----------------------
    # Evaluate
    # ----------------------
    y_pred = model.predict(X_test_vec)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    return model, vectorizer


if __name__ == "__main__":
    train()
