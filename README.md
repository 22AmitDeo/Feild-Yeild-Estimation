# Field Yield Estimation MVP

A production-style ML pipeline for estimating agricultural field yield using soil and weather variables — inspired by Agcurate's agritech approach.

---

## Problem Statement

Accurately predicting crop yield is critical for precision agriculture. Traditional linear models fail because the relationship between soil/weather variables and yield is highly non-linear. This project builds a complete ML pipeline that handles feature engineering, model comparison, explainability, and inference via a REST API.

---

## Dataset

| Feature | Description |
|---|---|
| `soil_ph` | Soil pH level (4.0 – 9.0) |
| `organic_matter` | Organic matter content (%) |
| `sand_pct` | Sand percentage in soil (%) |
| `temperature` | Average temperature (°C) |
| `humidity` | Relative humidity (%) |
| `rainfall` | Rainfall (mm) |
| `ndvi` | Normalized Difference Vegetation Index |
| `yield` | **Target** — crop yield (kg/ha) |

- 10,000 records, no missing values
- Yield range: ~3,500 – 5,000 kg/ha

---

## Approach

### Feature Engineering
Beyond raw features, the following are engineered:
- `rainfall × temperature` — combined climate stress
- `humidity × temperature` — heat-humidity interaction
- `organic_matter × rainfall` — nutrient availability under moisture
- `sand_pct × rainfall` — drainage effect
- `rainfall_to_temp_ratio` — moisture efficiency
- `soil_quality_index` — organic matter weighted by pH optimality

### Models Compared

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Regularized linear |
| Lasso Regression | Sparse linear |
| Random Forest | Tree ensemble |
| Gradient Boosting | Boosted trees |
| XGBoost | Optimized boosting |

### Evaluation Metrics
- **R²** — explained variance
- **RMSE** — root mean squared error
- **MAE** — mean absolute error

---

## Results

After training, metrics are saved to `results/metrics.json`. The best model (typically XGBoost or Gradient Boosting) is saved to `models/yield_model.pkl`.

Plots saved to `results/plots/`:
- `correlation_heatmap.png`
- `feature_importance_<model>.png`
- `pred_vs_actual_<model>.png`
- `residuals_<model>.png`
- `shap_summary_<model>.png`

---

## Project Structure

```
Feild-Yeild-Estimation/
├── data/
│   └── raw/train.csv
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
│   └── yield_model.pkl
├── results/
│   ├── plots/
│   └── metrics.json
├── api/
│   └── main.py
├── notebooks/
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the pipeline
```bash
python src/train_model.py
```

This will:
- Load and engineer features
- Train 6 models
- Save metrics to `results/metrics.json`
- Save plots to `results/plots/`
- Save the best model to `models/yield_model.pkl`

### 3. Start the API
```bash
uvicorn api.main:app --reload
```

### 4. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "soil_ph": 6.5,
    "organic_matter": 2.1,
    "sand_pct": 35,
    "temperature": 28,
    "humidity": 70,
    "rainfall": 120,
    "ndvi": 0.65
  }'
```

**Response:**
```json
{"predicted_yield": 4213.57}
```

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to explain individual predictions from the best tree-based model. The summary plot shows which features drive yield up or down across the test set.

---

## Tech Stack

- **Python 3.11**
- scikit-learn, XGBoost, SHAP
- FastAPI + Uvicorn
- Matplotlib, Seaborn
- Joblib
