import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re


def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/SK7here/Movie-Review-Sentiment-Analysis/master/IMDB-Dataset.csv"
    df = pd.read_csv(url)

    # Take a smaller subset for faster testing (optional, remove if you want full dataset)
    df = df.sample(n=1000, random_state=42)

    print("Dataset Info:")
    print(df['sentiment'].value_counts())
    print(df.isnull().sum())

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text

    df['cleaned_review'] = df['review'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_review'])
    y = df['sentiment'].map({'positive': 1, 'negative': 0})

    return X, y, tfidf