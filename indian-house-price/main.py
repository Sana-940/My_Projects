import logging
from datetime import datetime

 


import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
 
model = joblib.load("random_forest_house_rent.pkl")
app= FastAPI(title='House Rent Prediction API', description='This API predicts the rent of a house based on its features.', version='1.0.0')


logging.basicConfig(filename='app.log', level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s')# for simple logging configuration aka monitoring

class HouseFeatures(BaseModel): # expected input data model for the API
    BHK: float | None = None
    Size: float | None = None
    Bathroom: float | None = None
    Current_Floor: float | None = None
    Total_Floors: float | None = None
    Area_Locality: str
    Area_Type: str
    City: str
    Furnishing_Status: str
    Tenant_Preferred: str
    Point_of_Contact: str
 
    class Config:
        populate_by_name = True # so the model can be populated by field names, even if they are not in the same order as the input data    
 
 
class PredictionResponse(BaseModel): # the expected output data model for the API
    predicted_rent: float

@app.get("/health") #readiness checkpoint
def health():
    return {"status": "ok", "model_loaded": model is not None}
 

 
@app.post("/predict", response_model=PredictionResponse) 
#checkpoint for prediction, takes in the input data model and returns the output data model 
# if not successful returns an error message 422
def predict(features: HouseFeatures):
    input_dict = {
        "BHK": features.BHK,
        "Size": features.Size,
        "Bathroom": features.Bathroom,
        "Current Floor": features.Current_Floor,
        "Total Floors": features.Total_Floors,
        "Area Locality": features.Area_Locality,
        "Area Type": features.Area_Type,
        "City": features.City,
        "Furnishing Status": features.Furnishing_Status,
        "Tenant Preferred": features.Tenant_Preferred,
        "Point of Contact": features.Point_of_Contact,
    }
    input_df = pd.DataFrame([input_dict])#convert the input dictionary to a pandas DataFrame for prediction
 
    log_prediction = model.predict(input_df)[0]
    rent_prediction = float(np.expm1(log_prediction))
 
    logging.info(
        f"input={input_dict} predicted_rent={rent_prediction}"
    )
 
    return PredictionResponse(predicted_rent=rent_prediction) 