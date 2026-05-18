from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.utils.dataset_store import load_dataset_summary

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/prediction")
def prediction_page(request: Request):
    metadata = {}
    store = getattr(request.app.state, "model_store", None)
    if store and store.metadata:
        metadata = store.metadata

    dataset_summary = getattr(request.app.state, "dataset_summary", None)
    if not isinstance(dataset_summary, dict):
        dataset_summary = load_dataset_summary()
        if isinstance(dataset_summary, dict):
            request.app.state.dataset_summary = dataset_summary

    feature_columns = metadata.get("feature_columns") or (store.feature_columns if store else []) or []
    dataset_columns = dataset_summary.get("feature_columns") if isinstance(dataset_summary, dict) else None
    if dataset_columns:
        feature_columns = dataset_columns
        
    context = {"request": request, "feature_columns": feature_columns}
    return templates.TemplateResponse(request=request, name="prediction.html", context=context)