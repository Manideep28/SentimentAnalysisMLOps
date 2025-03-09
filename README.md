# Sentiment Analysis MLOps Project

## Overview
This project implements an end-to-end MLOps pipeline for sentiment analysis using the IMDB dataset.

## Data Preprocessing
- Loaded IMDB dataset
- Removed stopwords and punctuation
- Used TF-IDF vectorization with 5000 features
- Encoded sentiments as binary labels

## Model Selection
- Chose Logistic Regression for its simplicity and effectiveness
- Achieved ~85% accuracy on test set
- Saved model using Joblib

## Deployment
- Built REST API using FastAPI
- /predict endpoint accepts JSON input
- Containerized with Docker
- Basic logging implemented

## CI/CD
- GitHub Actions pipeline for automated testing and deployment
- Docker image building on push to main

## Monitoring
- MLflow for experiment tracking
- Basic logging for API requests


## Running Locally
1. Activate virtual env: `.\venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Train model: `python src/train.py`
   - Saves models to root directory
4. Run API: `uvicorn src.app:app --reload`
5. Test: `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "Great movie"}'`


## Future Improvements
- Add authentication
- Implement model versioning
- Set up Prometheus/Grafana monitoring