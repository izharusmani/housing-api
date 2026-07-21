import joblib
import numpy as np

MODEL_PATH = "model/house_price_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_single(data):
    features = np.array([
        [
            data.square_footage,
            data.bedrooms,
            data.bathrooms,
            data.year_built,
            data.lot_size,
            data.distance_to_city_center,
            data.school_rating
        ]
    ])

    prediction = model.predict(features)

    return float(prediction[0])


def predict_batch(data_list):
    features = []

    for item in data_list:
        features.append([
            item.square_footage,
            item.bedrooms,
            item.bathrooms,
            item.year_built,
            item.lot_size,
            item.distance_to_city_center,
            item.school_rating
        ])

    predictions = model.predict(np.array(features))

    return predictions.tolist()