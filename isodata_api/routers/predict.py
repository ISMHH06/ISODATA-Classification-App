from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from sklearn.manifold import TSNE

from app.utils.dataset_store import UPLOAD_DIR, load_dataset_summary
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
from isodata_api.services.predictor import build_cluster_space, run_pipeline

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


@router.get("/visualization", response_model=None)
def visualization(request: Request, sample: int | None = None, store: ModelStore = Depends(get_model_store)):
    _ensure_loaded(store)
    dataset_summary = getattr(request.app.state, "dataset_summary", None)
    if not isinstance(dataset_summary, dict):
        dataset_summary = load_dataset_summary()
        if isinstance(dataset_summary, dict):
            request.app.state.dataset_summary = dataset_summary

    if not dataset_summary or "dataset" not in dataset_summary:
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    dataset_name = str(dataset_summary["dataset"])
    dataset_path = UPLOAD_DIR / dataset_name
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    cache_key = (dataset_name, int(sample or 0), store.n_clusters, store.n_samples_train)
    cache = getattr(request.app.state, "tsne_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache.get("payload", {})

    df = pd.read_csv(dataset_path)
    if sample and sample > 0 and len(df.index) > sample:
        df = df.sample(sample, random_state=42)

    try:
        cluster_space = build_cluster_space(df, store)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    labels = store.iso_model.predict(cluster_space)
    n_points = int(cluster_space.shape[0])

    if n_points < 3:
        coords = cluster_space[:, :2]
        if coords.shape[1] < 2:
            coords = np.column_stack([np.arange(n_points), np.zeros(n_points)])
    else:
        perplexity = min(50, max(5, n_points // 10))
        if perplexity >= n_points:
            perplexity = max(1, n_points - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="random",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(cluster_space)

    points = [
        {
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "cluster_id": int(labels[idx]),
        }
        for idx in range(n_points)
    ]

    cluster_sizes: Dict[int, int] = {}
    for label in labels:
        key = int(label)
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1

    payload = {
        "points": points,
        "cluster_sizes": cluster_sizes,
        "n_points": n_points,
    }
    request.app.state.tsne_cache = {"key": cache_key, "payload": payload}
    return payload


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
