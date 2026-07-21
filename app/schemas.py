from pydantic import BaseModel
from typing import List


class HouseFeatures(BaseModel):
    square_footage: int
    bedrooms: int
    bathrooms: float
    year_built: int
    lot_size: int
    distance_to_city_center: float
    school_rating: float


class PredictionResponse(BaseModel):
    predicted_price: float


class BatchPredictionResponse(BaseModel):
    predictions: List[float]