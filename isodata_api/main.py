from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from isodata_api.models.model_loader import load_model_store
from isodata_api.routers.predict import router as predict_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(__file__).resolve().parent / "model_export"
    store = load_model_store(model_dir)
    app.state.model_store = store
    if store.model_loaded:
        logger.info("Model store ready with %s clusters", store.n_clusters)
    else:
        logger.error("Model store failed to load: %s", store.load_error)
    yield


app = FastAPI(
    title="ISODATA Credit Card Segmentation API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1", tags=["predict"])
