# ISODATA Credit Card Segmentation API

FastAPI service for credit card customer segmentation using a trained ISODATA clustering model.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your model artifacts in `isodata_api/model_export/`:

- `isodata_model.joblib`
- `scaler.joblib`
- `pca_cluster.joblib`
- `pca_2d.joblib`
- `feature_columns.json`
- `cluster_profiles.json`
- `model_metadata.json`
- `winsor_bounds.json` (optional)

Run the API:

```bash
uvicorn main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Example request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "BALANCE": 1200.0,
    "BALANCE_FREQUENCY": 0.9,
    "PURCHASES": 450.0,
    "ONEOFF_PURCHASES": 300.0,
    "INSTALLMENTS_PURCHASES": 150.0,
    "CASH_ADVANCE": 0.0,
    "PURCHASES_FREQUENCY": 0.6,
    "ONEOFF_PURCHASES_FREQUENCY": 0.5,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.4,
    "CASH_ADVANCE_FREQUENCY": 0.0,
    "CASH_ADVANCE_TRX": 0,
    "PURCHASES_TRX": 12,
    "CREDIT_LIMIT": 5000.0,
    "PAYMENTS": 600.0,
    "MINIMUM_PAYMENTS": 200.0,
    "PRC_FULL_PAYMENT": 0.3,
    "TENURE": 12
  }'
```
