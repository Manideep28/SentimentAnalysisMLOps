import mlflow


def setup_monitoring():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("sentiment_analysis")

    with mlflow.start_run():
        X, y, _ = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Log metrics
        y_pred = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.sklearn.log_model(model, "model")