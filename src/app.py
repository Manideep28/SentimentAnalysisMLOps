from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from pydantic import BaseModel
import joblib
import logging
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.pkl')
TFIDF_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file at {MODEL_PATH} or vectorizer file at {TFIDF_PATH} is missing. Train the model first.")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict_sentiment(input: TextInput):
    try:
        # Preprocess input
        cleaned_text = ' '.join([word for word in input.text.lower().split()
                                 if word not in stopwords.words('english')])
        vectorized_text = tfidf.transform([cleaned_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        logger.info(f"Prediction made for text: {input.text}")

        return {
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": float(max(probability))
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))