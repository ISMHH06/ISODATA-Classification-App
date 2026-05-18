import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from app.utils.dataset_store import load_dataset_summary, save_dataset_file, save_dataset_summary

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/dataset")
def dataset_page(request: Request):
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    dataset_summary = getattr(request.app.state, "dataset_summary", None)
    if not isinstance(dataset_summary, dict):
        dataset_summary = load_dataset_summary()
        if isinstance(dataset_summary, dict):
            request.app.state.dataset_summary = dataset_summary

    feature_columns = metadata.get("feature_columns") or (store.feature_columns if store else []) or []
    dataset_columns = dataset_summary.get("feature_columns") if isinstance(dataset_summary, dict) else None
    if dataset_columns:
        feature_columns = dataset_columns
    hyperparams = metadata.get("hyperparameters") if isinstance(metadata, dict) else {}
    context = {
        "request": request,
        "metadata": metadata,
        "n_rows": (dataset_summary or {}).get("rows") or metadata.get("n_samples_train") or 0,
        "n_features": len(feature_columns),
        "missing_values": (dataset_summary or {}).get("missing_values") or 0,
        "duplicate_rows": (dataset_summary or {}).get("duplicate_rows") or 0,
        "hyperparams": hyperparams or {},
        "dataset_summary": dataset_summary or {},
    }
    return templates.TemplateResponse(request=request, name="dataset.html", context=context)


@router.post("/dataset/summary")
async def dataset_summary(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename
    if not filename or not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Failed to read CSV file") from exc

    rows, cols = df.shape
    dataset_name = filename
    feature_columns = df.columns.tolist()
    numeric_df = df.select_dtypes(include="number")

    corr_payload = None
    if not numeric_df.empty:
        corr_df = numeric_df.corr(numeric_only=True)
        corr_columns = corr_df.columns.tolist()[:12]
        corr_df = corr_df.loc[corr_columns, corr_columns].round(4).fillna(0.0)
        corr_payload = {
            "columns": corr_columns,
            "matrix": corr_df.to_numpy().tolist(),
        }

    distributions = None
    if not numeric_df.empty:
        dist_columns = numeric_df.columns.tolist()[:2]
        bins = {}
        counts = {}
        for column in dist_columns:
            series = numeric_df[column].dropna()
            if series.empty:
                continue
            hist_counts, hist_edges = np.histogram(series.to_numpy(), bins=10)
            bins[column] = [float(value) for value in hist_edges.tolist()]
            counts[column] = [int(value) for value in hist_counts.tolist()]
        if bins and counts:
            distributions = {
                "columns": dist_columns,
                "bins": bins,
                "counts": counts,
            }
    dataset_name = file.filename or ""
    feature_columns = df.columns.tolist()
    summary = {
        "rows": int(rows),
        "columns": int(cols),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": feature_columns,
        "dataset": dataset_name,
        "feature_columns": feature_columns,
        "correlation": corr_payload,
        "distributions": distributions,
    }
    summary_store = {
        **summary,
        "n_samples_train": int(rows),
    }
    request.app.state.dataset_summary = summary_store
    save_dataset_file(dataset_name, content)
    save_dataset_summary(summary_store)
    return JSONResponse(content=summary)