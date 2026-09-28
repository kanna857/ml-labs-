# MLOps Lab – 12 Experiments

GitHub-ready implementation based on the supplied MLOps Lab Manual.

## Experiments
1. Git repository initialization and version control
2. ML project folder structure
3. Model serialization using Joblib
4. MLflow experiment tracking
5. MLflow experiment comparison
6. Scikit-learn preprocessing pipeline
7. Reusable end-to-end ML pipeline
8. GitHub Actions CI
9. FastAPI model serving
10. FastAPI validation and error handling
11. Docker containerization
12. Monitoring and data drift detection

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

## Run experiments

```bash
python experiments/exp03_serialization.py
python experiments/exp04_mlflow_tracking.py
python experiments/exp05_mlflow_comparison.py
python experiments/exp06_pipeline.py
python experiments/exp07_reusable_pipeline.py
python experiments/exp12_drift.py
```

For FastAPI:
```bash
python experiments/exp07_reusable_pipeline.py
uvicorn experiments.exp09_api:app --reload
```

Open:
`http://127.0.0.1:8000/docs`

For Docker:
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

For MLflow:
```bash
mlflow ui
```
Then open `http://127.0.0.1:5000`.

For tests:
```bash
pytest -q
```

## GitHub push

```bash
git init
git add .
git commit -m "Add MLOps lab experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mlops-lab.git
git push -u origin main
```

