import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datetime import datetime

def predict_test(data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs") -> None:
   
    model_path = os.path.join(models_dir, "titanic_pipeline.joblib")
    test_csv = os.path.join(data_dir, "test.csv")
    targets_csv = os.path.join(data_dir, "gender_submission.csv")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Please run training first.")
        return

    print(f"[TEST] Loading pipeline: {model_path}")
    pipe = joblib.load(model_path)
    

    df = pd.read_csv(test_csv)
    passenger_ids = df["PassengerId"].copy()
    X_test = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

   
    print(f"[TEST] Loading actual targets: {targets_csv}")
    targets_df = pd.read_csv(targets_csv)
    y_true = targets_df["Survived"].values

    print("[TEST] Predicting...")
    preds = pipe.predict(X_test)

   
    accuracy = accuracy_score(y_true, preds)
    

    print("\n" + "="*50)
    print(f"TEST ACCURACY: {accuracy:.4f}")
    print("="*50)

    
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, "predictions.csv")
    pd.DataFrame({"PassengerId": passenger_ids, "Survived": preds}).to_csv(out_path, index=False)
    print(f"[TEST] Wrote predictions to {out_path}")