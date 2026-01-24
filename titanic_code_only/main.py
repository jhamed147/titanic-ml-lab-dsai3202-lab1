import argparse
import joblib
import os
import pandas as pd
from fastapi import FastAPI
from src.train import train_model
from src.test import predict_test
from src.predict import Passenger 

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival using a scikit-learn model saved with joblib.",
    version="1.0.0",
)

MODEL_PATH = "models/titanic_pipeline.joblib"

@app.get("/")
def read_root():
    return {"message": "Titanic API is online"}

@app.post("/predict")
def predict_survival(passenger: Passenger):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found"}

    model = joblib.load(MODEL_PATH)
    input_data = pd.DataFrame([passenger.dict()])
    prediction = model.predict(input_data)[0]
    
    status = "Survived" if prediction == 1 else "Did not survive"
    return {
        "prediction": int(prediction),
        "status": status
    }

def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'])
    args = parser.parse_args()

    if args.mode == 'train':
        train_model('data/train.csv')
    elif args.mode == 'test':
        predict_test(data_dir="data", models_dir="models", outputs_dir="outputs")
    else:
        print("Use: uvicorn main:app --reload to start the API")

if __name__ == "__main__":
    run_cli()