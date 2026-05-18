import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/dataset")
def dataset_page(request: Request):
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    feature_columns = metadata.get("feature_columns") or (store.feature_columns if store else []) or []
    hyperparams = metadata.get("hyperparameters") if isinstance(metadata, dict) else {}
    context = {
        "request": request,
        "metadata": metadata,
        "n_rows": metadata.get("n_samples_train") or 0,
        "n_features": len(feature_columns),
        "missing_values": 0,
        "duplicate_rows": 0,
        "hyperparams": hyperparams or {},
    }
    return templates.TemplateResponse(name="dataset.html", context=context)


@router.post("/dataset/summary")
async def dataset_summary(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Failed to read CSV file") from exc

    rows, cols = df.shape
    summary = {
        "rows": int(rows),
        "columns": int(cols),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": df.columns.tolist(),
    }
    return JSONResponse(content=summary)