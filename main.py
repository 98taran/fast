from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


class Input(BaseModel):
    Pregnancies : int
    Glucose : int
    BloodPresure:int
    SkinThickness:int
    Insulin : int
    BMI:float
    DiabetesPedigreeFunction : float
    Age : int


@app.get("/")

def check():
    return {"message":" working fine"}


@app.post("/diabetic")
def predict(data : Input):


    features= [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]

    features_scaled = scaler.transform(features)
    prediction =model.predict(features_scaled)

    
    probability =model.predict_proba(features_scaled)

    if prediction[0] == 1:
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return {
        "prediction": result,

    }