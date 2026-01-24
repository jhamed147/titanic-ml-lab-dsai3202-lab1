import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.data.preprocessing import build_preprocessor

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
    y = df["Survived"]

    num_cols = ["Age", "Fare", "SibSp", "Parch"]
    cat_cols = ["Pclass", "Sex", "Embarked"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("[TRAIN] Training model...")
    full_pipeline.fit(X, y)

    os.makedirs('models', exist_ok=True)
    joblib.dump(full_pipeline, 'models/titanic_pipeline.joblib')
    print("[TRAIN] Model saved as models/titanic_pipeline.joblib")