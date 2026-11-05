from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from services.service import PredictionService
import pandas as pd
import os

router = APIRouter()

artifact_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    'artifact',
    'linear_regression_model.pkl'
)

artifact_path = os.path.abspath(artifact_path)

prediction_service = PredictionService(model_path=artifact_path)

@router.post("/predict")
async def predict(
    SquareFeet: int = Form(...),
    Bedrooms: int = Form(...),
    Bathrooms: int = Form(...),
    Neighborhood: str = Form(...),
    YearBuilt: int = Form(...)

):
    # Create a DataFrame from the input data
    input_features = [{
        "SquareFeet": SquareFeet,
        "Bedrooms": Bedrooms,
        "Bathrooms": Bathrooms,
        "Neighborhood": Neighborhood,
        "YearBuilt": YearBuilt
    }]

    features = pd.DataFrame(input_features)
    predicted_price = prediction_service.predict(features)
    return JSONResponse(content={"price": float(predicted_price)})
    
