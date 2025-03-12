FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"
COPY . .
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]