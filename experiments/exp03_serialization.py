import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.DataFrame({
    "area": [800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [2, 2, 3, 3, 4, 4],
    "price": [40, 50, 60, 75, 90, 100]
})

X = data[["area", "bedrooms"]]
y = data["price"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "house_price_model.joblib")
loaded_model = joblib.load("house_price_model.joblib")

new_house = pd.DataFrame({"area": [1600], "bedrooms": [3]})
prediction = loaded_model.predict(new_house)[0]

print("Predicted price:", round(prediction, 2), "lakhs")
print("Model file created: house_price_model.joblib")
