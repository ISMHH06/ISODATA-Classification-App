from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
def home(request: Request):
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    feature_columns = metadata.get("feature_columns") or (store.feature_columns if store else []) or []
    context = {
        "request": request,
        "metadata": metadata,
        "n_clusters": metadata.get("n_clusters_final") or (store.n_clusters if store else 0),
        "n_features": len(feature_columns),
    }
    return templates.TemplateResponse(name="home.html", context=context)