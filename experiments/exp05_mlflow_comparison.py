import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Model_Comparison")

models = [
    ("RF_50", RandomForestRegressor(n_estimators=50, random_state=42)),
    ("RF_150", RandomForestRegressor(n_estimators=150, random_state=42)),
    ("GBR", GradientBoostingRegressor(random_state=42))
]

for name, model in models:
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(name, "MAE =", round(mae, 3), "R2 =", round(r2, 3))
