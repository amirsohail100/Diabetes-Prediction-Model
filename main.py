import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import Optional,Literal
from joblib import load
from keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = load("columns.pkl")
model = load_model("model.h5")
preprocessing = load("preprocessing.pkl")

class Data(BaseModel):
    pregnancies: int = Field(..., ge=10, le=100)
    glucose: float = Field(..., ge=0, le=24)
    bloodpressure: float = Field(..., ge=0, le=24)
    skinthickness: float = Field(..., ge=0, le=24)
    insulin: float = Field(..., ge=0, le=24)
    bmi: float = Field(..., ge=0, le=24)
    diabetespedigreefunction: float = Field(..., ge=0, le=24)
    age: int = Field(..., ge=10, le=100)

class PredictionResponse(BaseModel):
    predicted: Literal["Yes","No"]

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED"}

@app.post("/predict")
def predict(data: Data):

    input_row = pd.DataFrame([{
        "Pregnancies":data.pregnancies,
        "Glucose":data.glucose,
        "BloodPressure":data.bloodpressure,
        "SkinThickness":data.skinthickness,
        "Insulin":data.insulin,
        "BMI":data.bmi,
        "DiabetesPedigreeFunction":data.diabetespedigreefunction,
        "Age":data.age
    }])

    preprocessing_df = preprocessing.transform(input_row)[0]
    prediction = model.predict(preprocessing_df)

    if prediction == 1:
        output = "Yes"
    else:
        output = "No"
    return PredictionResponse(predicted_mental_health_score=output)