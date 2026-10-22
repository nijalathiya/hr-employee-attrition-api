

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="HR Employee Attrition API")

model = joblib.load("hr_attrition_model.pkl")

@app.get("/")
def home():
    return {"message": "HR Employee Attrition API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0]
    classes = list(model.classes_)
    leave_index = classes.index(1) if 1 in classes else None

    leave_probability = (
        float(probability[leave_index])
        if leave_index is not None
        else None
    )

    return {
        "prediction": str(prediction),
        "leave_probability": leave_probability
    }
