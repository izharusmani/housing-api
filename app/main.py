from typing import List

from fastapi import FastAPI

from app.schemas import (
    HouseFeatures,
    PredictionResponse,
    BatchPredictionResponse,
)

from app.predictor import (
    predict_single,
    predict_batch,
)

app = FastAPI(
    title="Housing Price Prediction API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Housing Price Prediction API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "Linear Regression",
        "version": "1.0",
        "target": "price",
        "features": [
            "square_footage",
            "bedrooms",
            "bathrooms",
            "year_built",
            "lot_size",
            "distance_to_city_center",
            "school_rating",
        ],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(data: HouseFeatures):

    prediction = predict_single(data)

    return {
        "predicted_price": prediction
    }


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def predict_multiple(data: List[HouseFeatures]):

    predictions = predict_batch(data)

    return {
        "predictions": predictions
    }