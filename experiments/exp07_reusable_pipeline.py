import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_pipeline():
    data = pd.DataFrame({
        "area": [800,1000,1200,1500,1800,2000,2200,2500],
        "bedrooms": [2,2,3,3,4,4,4,5],
        "price": [40,50,60,75,90,100,110,125]
    })

    X = data[["area", "bedrooms"]]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    joblib.dump(pipeline, "reusable_house_pipeline.joblib")

    print("Training complete")
    print("MAE:", round(mae, 3))
    print("Saved: reusable_house_pipeline.joblib")

if __name__ == "__main__":
    train_pipeline()
