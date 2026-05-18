from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/dashboard")
def dashboard_page(request: Request):
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    metrics = metadata.get("metrics") if isinstance(metadata, dict) else {}
    n_clusters = metadata.get("n_clusters_final") or (store.n_clusters if store else 0)
    n_samples = metadata.get("n_samples_train") or (store.n_samples_train if store else 0)

    profiles = store.cluster_profiles if store and store.cluster_profiles else {}
    sizes = store.cluster_sizes if store and store.cluster_sizes else {}
    clusters = []
    for cluster_key in sorted(profiles.keys(), key=lambda value: int(value)):
        cluster_id = int(cluster_key)
        size = int(sizes.get(cluster_id, 0)) if sizes else 0
        pct = (float(size) / float(n_samples) * 100.0) if n_samples else 0.0
        clusters.append({"id": cluster_id, "size": size, "pct": pct})

    context = {
        "request": request,
        "metrics": metrics or {},
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "clusters": clusters,
    }
    return templates.TemplateResponse(name="dashboard.html", context=context)