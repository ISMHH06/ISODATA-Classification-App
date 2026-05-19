from typing import Dict, List

from pydantic import BaseModel, Field


class ClientInput(BaseModel):
    BALANCE: float = Field(..., ge=0)
    BALANCE_FREQUENCY: float = Field(..., ge=0, le=1)
    PURCHASES: float = Field(..., ge=0)
    ONEOFF_PURCHASES: float = Field(..., ge=0)
    INSTALLMENTS_PURCHASES: float = Field(..., ge=0)
    CASH_ADVANCE: float = Field(..., ge=0)
    PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1)
    ONEOFF_PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1)
    PURCHASES_INSTALLMENTS_FREQUENCY: float = Field(..., ge=0, le=1)
    CASH_ADVANCE_FREQUENCY: float = Field(..., ge=0, le=1)
    CASH_ADVANCE_TRX: int = Field(..., ge=0)
    PURCHASES_TRX: float = Field(..., ge=0)
    CREDIT_LIMIT: float | None = Field(None, ge=0)
    PAYMENTS: float = Field(..., ge=0)
    MINIMUM_PAYMENTS: float | None = Field(None, ge=0)
    PRC_FULL_PAYMENT: float = Field(..., ge=0, le=1)
    TENURE: float = Field(..., ge=0)


class PcaCoords(BaseModel):
    pc1: float
    pc2: float


class EmbedCoords(BaseModel):
    x: float
    y: float


class PredictionResponse(BaseModel):
    segment_id: int
    n_clusters: int
    pca_coords: PcaCoords
    cluster_profile: Dict[str, float]
    segment_size_pct: float


class ClusterProfile(BaseModel):
    cluster_id: int
    n_clients_train: int
    pct_portfolio: float
    profile: Dict[str, float]
    tsne_coords: EmbedCoords | None = None


class BatchRequest(BaseModel):
    clients: List[ClientInput] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    n_processed: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_clusters: int
