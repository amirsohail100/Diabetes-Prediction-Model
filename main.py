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

# Model aur Preprocessing objects load kar rahe hain
COLUMNS = load("columns.pkl")
model = load_model("model.h5")
preprocessing = load("preprocessing.pkl")

# PIMA Diabetes Dataset ke real values ke hisaab se Pydantic Model
class Data(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant")
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    bloodpressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skinthickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body mass index (weight in kg/(height in m)^2)")
    diabetespedigreefunction: float = Field(..., ge=0, le=3.0, description="Diabetes pedigree function")
    age: int = Field(..., ge=1, le=120, description="Age in years")

class PredictionResponse(BaseModel):
    predicted: Literal['Yes', 'No']

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED - Diabetes Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: Data):
    # Input data ko DataFrame mein convert karna (Changed "pregnancies" to "Pregnancies")
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
    
    # Preprocessing aur Prediction
    preprocessing_df = preprocessing.transform(input_row)
    prediction_prob = model.predict(preprocessing_df)[0][0]
    
    # Thresholding (0.5 ke hisaab se binary conversion)
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    output = "Yes" if prediction == 1 else "No"
        
    return PredictionResponse(predicted=output)
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