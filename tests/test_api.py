import pytest
import requests


def test_predict_endpoint():
    url = "http://localhost:8000/predict"
    payload = {"text": "This is a great movie!"}

    response = requests.post(url, json=payload)
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()