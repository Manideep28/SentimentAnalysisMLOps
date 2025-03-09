from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from pydantic import BaseModel
import joblib
import logging
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Ensure the correct paths
model_path = os.path.join(os.path.dirname(__file__), "../sentiment_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "../tfidf_vectorizer.pkl")

# Check if files exist before loading
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file is missing. Train the model first.")


# Load models
model = joblib.load('../sentiment_model.pkl')
tfidf = joblib.load('../tfidf_vectorizer.pkl')


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