import pandas as pd

df = pd.read_csv("data/House Price Dataset.csv")

print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 3: Select Features (X)
X = df[
    [
        "square_footage",
        "bedrooms",
        "bathrooms",
        "year_built",
        "lot_size",
        "distance_to_city_center",
        "school_rating"
    ]
]

# Step 4: Select Target (y)
y = df["price"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:10])

from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

r2 = r2_score(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print(r2)
print(rmse)

import joblib

joblib.dump(model, "model/house_price_model.pkl")