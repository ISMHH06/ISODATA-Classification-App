from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import pandas as pd

from isodata_api.models.model_loader import ModelStore

RAW_FEATURES = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

DROP_COLUMNS = [
    "ONEOFF_PURCHASES",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "CASH_ADVANCE_FREQUENCY",
    "PURCHASES_FREQUENCY",
]

IMPUTE_COLUMNS = ["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]

EPSILON = 1e-6


def _extract_bounds(bounds: Any) -> Tuple[float, float] | None:
    if bounds is None:
        return None

    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        return float(bounds[0]), float(bounds[1])

    if isinstance(bounds, dict):
        lower = None
        upper = None
        for key in ("p1", "P1", "lower", "min", "low"):
            if key in bounds:
                lower = bounds[key]
                break
        for key in ("p99", "P99", "upper", "max", "high"):
            if key in bounds:
                upper = bounds[key]
                break
        if lower is not None and upper is not None:
            return float(lower), float(upper)

    return None


def _apply_winsorization(df: pd.DataFrame, winsor_bounds: Dict[str, Any]) -> pd.DataFrame:
    for column in RAW_FEATURES:
        bounds = winsor_bounds.get(column)
        parsed = _extract_bounds(bounds)
        if parsed is None:
            continue
        lower, upper = parsed
        df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def _impute_value(feature: str, store: ModelStore) -> float:
    if feature in IMPUTE_COLUMNS:
        if store.imputation_values and feature in store.imputation_values:
            return float(store.imputation_values[feature])
        raise ValueError(f"Missing required field: {feature}")
    raise ValueError(f"Missing required field: {feature}")


def _coerce_value(feature: str, value: Any, store: ModelStore) -> float:
    if value is None:
        return _impute_value(feature, store)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for {feature}") from exc
    if math.isnan(number):
        return _impute_value(feature, store)
    return number


def _segment_size_pct(cluster_id: int, store: ModelStore) -> float:
    if store.cluster_sizes and store.n_samples_train > 0:
        size = store.cluster_sizes.get(cluster_id)
        if size is not None:
            return float(size) / float(store.n_samples_train) * 100.0
    return 0.0


def run_pipeline(client_data: Dict[str, Any], store: ModelStore) -> Dict[str, Any]:
    if not store.model_loaded:
        raise RuntimeError("Model not loaded")

    if (
        store.scaler is None
        or store.pca_cluster is None
        or store.pca_2d is None
        or store.iso_model is None
    ):
        raise RuntimeError("Model artifacts not loaded")

    scaler = store.scaler
    pca_cluster = store.pca_cluster
    pca_2d = store.pca_2d
    iso_model = store.iso_model

    missing = [
        feature
        for feature in RAW_FEATURES
        if feature not in client_data and feature not in IMPUTE_COLUMNS
    ]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    row = {
        feature: _coerce_value(feature, client_data.get(feature), store)
        for feature in RAW_FEATURES
    }
    df = pd.DataFrame([row], columns=RAW_FEATURES)

    if store.winsor_bounds:
        df = _apply_winsorization(df, store.winsor_bounds)

    df["TOTAL_ONEOFF_RATIO"] = df["ONEOFF_PURCHASES"] / (df["PURCHASES"] + EPSILON)
    df["INSTALLMENT_DOMINANCE"] = (
        df["PURCHASES_INSTALLMENTS_FREQUENCY"] / (df["PURCHASES_FREQUENCY"] + EPSILON)
    )
    df["CASH_ADVANCE_INTENSITY"] = df["CASH_ADVANCE"] / (df["CASH_ADVANCE_FREQUENCY"] + EPSILON)
    df["PAYMENT_TO_CREDIT_RATIO"] = df["PAYMENTS"] / (df["CREDIT_LIMIT"] + EPSILON)
    df["PAYMENT_TO_BALANCE_RATIO"] = df["PAYMENTS"] / (df["BALANCE"] + EPSILON)

    df = df.drop(columns=DROP_COLUMNS)

    feature_columns = store.feature_columns
    if not feature_columns:
        raise RuntimeError("Feature columns not loaded")

    ordered = df[feature_columns]
    features = ordered.to_numpy()

    scaled = scaler.transform(features)
    cluster_space = pca_cluster.transform(scaled)
    cluster_id = int(iso_model.predict(cluster_space)[0])

    coords = pca_2d.transform(scaled)[0]
    cluster_profiles = store.cluster_profiles or {}
    profile = cluster_profiles.get(str(cluster_id), {})

    return {
        "segment_id": cluster_id,
        "n_clusters": store.n_clusters,
        "pca_coords": {"pc1": float(coords[0]), "pc2": float(coords[1])},
        "cluster_profile": profile,
        "segment_size_pct": _segment_size_pct(cluster_id, store),
    }
