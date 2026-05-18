from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.utils.dataset_store import load_dataset_summary

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
def home(request: Request):
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    dataset_summary = getattr(request.app.state, "dataset_summary", None)
    if not isinstance(dataset_summary, dict):
        dataset_summary = load_dataset_summary()
        if isinstance(dataset_summary, dict):
            request.app.state.dataset_summary = dataset_summary

    if isinstance(dataset_summary, dict):
        metadata = {**metadata, **dataset_summary}
    feature_columns = metadata.get("feature_columns") or (store.feature_columns if store else []) or []
    context = {
        "request": request,
        "metadata": metadata,
        "n_clusters": metadata.get("n_clusters_final") or (store.n_clusters if store else 0),
        "n_features": len(feature_columns),
    }
    return templates.TemplateResponse(request=request, name="home.html", context=context)