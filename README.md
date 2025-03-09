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

## Running the Project
1. Install requirements: `pip install -r requirements.txt`
2. Train model: `python train.py`
3. Run API: `uvicorn app:app --reload`
4. Test with: `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "great movie"}'`

## Future Improvements
- Add authentication
- Implement model versioning
- Set up Prometheus/Grafana monitoring