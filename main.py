import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from joblib import load
from tensorflow.keras.models import load_model

app = FastAPI(title="Diabetes Prediction API")

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
    pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    bloodpressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skinthickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body mass index (weight in kg/(height in m)^2)")
    diabetespedigreefunction: float = Field(..., ge=0, le=3.0, description="Diabetes pedigree function")
    age: int = Field(..., ge=1, le=120, description="Age in years")

# Professional response structure with Risk Label and Probability
class PredictionResponse(BaseModel):
    prediction_label: Literal['High Risk', 'Low Risk']
    diabetes_probability: float = Field(..., description="Confidence score of the prediction")

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED - Diabetes Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: Data):
    input_row = pd.DataFrame([{
        "Pregnancies": data.pregnancies,
        "Glucose": data.glucose,
        "BloodPressure": data.bloodpressure,
        "SkinThickness": data.skinthickness,
        "Insulin": data.insulin,
        "BMI": data.bmi,
        "DiabetesPedigreeFunction": data.diabetespedigreefunction,
        "Age": data.age
    }])
    
    # Preprocessing and Prediction
    preprocessing_df = preprocessing.transform(input_row)
    prediction_prob = float(model.predict(preprocessing_df)[0][0])
    
    # Thresholding
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    # Professional Labels & Confidence calculation
    if prediction == 1:
        label = "High Risk"
        confidence = prediction_prob
    else:
        label = "Low Risk"
        confidence = 1.0 - prediction_prob
        
    return PredictionResponse(
        prediction_label=label,
        diabetes_probability=round(confidence, 4)
    )