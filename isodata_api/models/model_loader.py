from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from fastapi import Request

from models.isodata import ISODATA

logger = logging.getLogger(__name__)


@dataclass
class ModelStore:
    iso_model: Optional[Any] = None
    scaler: Optional[Any] = None
    pca_cluster: Optional[Any] = None
    pca_2d: Optional[Any] = None
    feature_columns: Optional[list[str]] = None
    cluster_profiles: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Optional[Dict[str, Any]] = None
    winsor_bounds: Optional[Dict[str, Any]] = None
    cluster_sizes: Optional[Dict[int, int]] = None
    n_clusters: int = 0
    n_samples_train: int = 0
    model_loaded: bool = False
    load_error: Optional[str] = None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _extract_cluster_sizes(iso_model: Any, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[int, int]]:
    sizes: Any = None
    if hasattr(iso_model, "cluster_sizes_"):
        sizes = getattr(iso_model, "cluster_sizes_")
    elif hasattr(iso_model, "cluster_counts_"):
        sizes = getattr(iso_model, "cluster_counts_")
    elif metadata and "cluster_sizes" in metadata:
        sizes = metadata["cluster_sizes"]
    elif metadata and "cluster_counts" in metadata:
        sizes = metadata["cluster_counts"]

    if sizes is None:
        return None

    if isinstance(sizes, dict):
        return {int(key): int(value) for key, value in sizes.items()}

    if isinstance(sizes, (list, tuple)):
        return {int(index): int(value) for index, value in enumerate(sizes)}

    return None


def load_model_store(model_dir: Path) -> ModelStore:
    store = ModelStore()
    try:
        setattr(sys.modules["__main__"], "ISODATA", ISODATA)
        iso_model = joblib.load(model_dir / "isodata_model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        pca_cluster = joblib.load(model_dir / "pca_cluster.joblib")
        pca_2d = joblib.load(model_dir / "pca_2d.joblib")
        feature_columns = _load_json(model_dir / "feature_columns.json")
        cluster_profiles = _load_json(model_dir / "cluster_profiles.json")
        metadata = _load_json(model_dir / "model_metadata.json")

        winsor_path = model_dir / "winsor_bounds.json"
        winsor_bounds = _load_json(winsor_path) if winsor_path.exists() else None

        n_clusters = getattr(iso_model, "n_clusters_", None)
        if n_clusters is None:
            n_clusters = len(cluster_profiles) if isinstance(cluster_profiles, dict) else 0

        n_samples_train = int(metadata.get("n_samples_train", 0)) if isinstance(metadata, dict) else 0
        cluster_sizes = _extract_cluster_sizes(iso_model, metadata)

        store = ModelStore(
            iso_model=iso_model,
            scaler=scaler,
            pca_cluster=pca_cluster,
            pca_2d=pca_2d,
            feature_columns=list(feature_columns),
            cluster_profiles=cluster_profiles,
            metadata=metadata,
            winsor_bounds=winsor_bounds,
            cluster_sizes=cluster_sizes,
            n_clusters=int(n_clusters),
            n_samples_train=n_samples_train,
            model_loaded=True,
        )
        logger.info("Model artifacts loaded from %s", model_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load model artifacts: %s", exc)
        store.model_loaded = False
        store.load_error = str(exc)

    return store


def get_model_store(request: Request) -> ModelStore:
    store = getattr(request.app.state, "model_store", None)
    if store is None:
        return ModelStore(model_loaded=False, load_error="Model store not initialized")
    return store
