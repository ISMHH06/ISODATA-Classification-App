from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request

from app.utils.dataset_store import load_dataset_summary
from fastapi.responses import JSONResponse

from isodata_api.models.model_loader import ModelStore, get_model_store
from isodata_api.schemas.schemas import (
    BatchRequest,
    BatchResponse,
    ClientInput,
    ClusterProfile,
    HealthResponse,
    PredictionResponse,
)
from isodata_api.services.predictor import run_pipeline

router = APIRouter()


def _ensure_loaded(store: ModelStore) -> None:
    if not store.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")


def _cluster_pct(cluster_id: int, store: ModelStore) -> float:
    if store.cluster_sizes and store.n_samples_train > 0:
        size = store.cluster_sizes.get(cluster_id, 0)
        return float(size) / float(store.n_samples_train) * 100.0
    return 0.0


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: ClientInput, store: ModelStore = Depends(get_model_store)) -> PredictionResponse:
    _ensure_loaded(store)
    try:
        result = run_pipeline(payload.model_dump(), store)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.post("/batch", response_model=BatchResponse)
def batch_predict(request: BatchRequest, store: ModelStore = Depends(get_model_store)) -> BatchResponse:
    _ensure_loaded(store)
    predictions = [PredictionResponse(**run_pipeline(client.model_dump(), store)) for client in request.clients]
    return BatchResponse(predictions=predictions, n_processed=len(predictions))


@router.get("/clusters", response_model=List[ClusterProfile])
def clusters(store: ModelStore = Depends(get_model_store)) -> List[ClusterProfile]:
    _ensure_loaded(store)
    profiles = store.cluster_profiles or {}
    results: List[ClusterProfile] = []
    for cluster_key in sorted(profiles.keys(), key=lambda value: int(value)):
        cluster_id = int(cluster_key)
        n_clients = 0
        if store.cluster_sizes:
            n_clients = int(store.cluster_sizes.get(cluster_id, 0))
        results.append(
            ClusterProfile(
                cluster_id=cluster_id,
                n_clients_train=n_clients,
                pct_portfolio=_cluster_pct(cluster_id, store),
                profile=profiles[cluster_key],
            )
        )
    return results


@router.get("/metadata")
def metadata(request: Request, store: ModelStore = Depends(get_model_store)) -> dict:
    _ensure_loaded(store)
    base = store.metadata or {}
    dataset_summary = getattr(request.app.state, "dataset_summary", None)
    if not isinstance(dataset_summary, dict):
        dataset_summary = load_dataset_summary()
        if isinstance(dataset_summary, dict):
            request.app.state.dataset_summary = dataset_summary
    if isinstance(dataset_summary, dict):
        return {**base, **dataset_summary}
    return base


@router.get("/health", response_model=HealthResponse)
def health(store: ModelStore = Depends(get_model_store)) -> JSONResponse | HealthResponse:
    payload = HealthResponse(
        status="ok",
        model_loaded=store.model_loaded,
        n_clusters=store.n_clusters,
    )
    if not store.model_loaded:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload
