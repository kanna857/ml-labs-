import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

data = pd.DataFrame({
    "area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "bedrooms": [2, 2, 3, 3, 4, 4, 4, 5],
    "city": ["A", "A", "B", "B", "A", "B", "B", "A"],
    "price": [40, 50, 60, 75, 90, 100, 110, 125]
})

X = data.drop(columns="price")
y = data["price"]

numeric = ["area", "bedrooms"]
categorical = ["city"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

new_house = pd.DataFrame({
    "area": [1600],
    "bedrooms": [3],
    "city": ["A"]
})

print("Prediction:", round(pipeline.predict(new_house)[0], 2), "lakhs")
