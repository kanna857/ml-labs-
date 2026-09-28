from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")
model = joblib.load("reusable_house_pipeline.joblib")

class House(BaseModel):
    area: float = Field(gt=0, description="Area must be positive")
    bedrooms: int = Field(ge=1, le=20, description="Bedrooms must be 1-20")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(house: House):
    data = pd.DataFrame([{
        "area": house.area,
        "bedrooms": house.bedrooms
    }])
    prediction = model.predict(data)[0]
    return {
        "status": "success",
        "predicted_price_lakhs": round(float(prediction), 2)
    }
