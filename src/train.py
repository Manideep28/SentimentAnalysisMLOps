from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from preprocessing import load_and_preprocess_data  # Changed from src.preprocessing

from sklearn.model_selection import train_test_split


def train_model():
    X, y, tfidf = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, '../sentiment_model.pkl')
    joblib.dump(tfidf, '../tfidf_vectorizer.pkl')

    return model, tfidf


if __name__ == "__main__":
    train_model()