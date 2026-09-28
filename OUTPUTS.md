# Expected Outputs

These outputs are based on the supplied lab manual and the included sample data. MLflow metric values are computed with `random_state=42`.

## Experiment 3
```text
Predicted price: 80.0 lakhs
Model file created: house_price_model.joblib
```

## Experiment 4
```text
Run logged successfully
MAE: 43.562
R2: 0.456
```

## Experiment 5
```text
RF_50 MAE = 44.765 R2 = 0.425
RF_150 MAE = 44.377 R2 = 0.435
GBR MAE = 44.603 R2 = 0.453
```

## Experiment 6
The script prints the prediction for the sample house. The exact value can vary with library versions.

## Experiment 7
The script prints:
```text
Training complete
MAE: <value>
Saved: reusable_house_pipeline.joblib
```

## Experiment 8
GitHub Actions should show:
```text
✓ Checkout code
✓ Set up Python
✓ Install dependencies
✓ Run tests
Process completed successfully.
```

## Experiment 9/10
Request:
```json
{"area":1600,"bedrooms":3}
```
Response shape:
```json
{
  "status": "success",
  "predicted_price_lakhs": 80.0
}
```

Invalid example:
```json
{"area":-10,"bedrooms":0}
```
Expected HTTP 422 validation error.

## Experiment 11
```text
Successfully tagged house-price-api:latest
Container started on port 8000
```

## Experiment 12
```text
Training mean: 1100.0
Live mean: 2100.0
Mean difference: 1000.0
Drift threshold: 400.0
WARNING: Possible data drift detected
```
