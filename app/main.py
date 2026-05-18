from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import dashboard, dataset, export, home, prediction
from isodata_api.models.model_loader import load_model_store
from isodata_api.routers.predict import router as predict_router

API_DIR = Path(__file__).resolve().parents[1] / "isodata_api"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	model_dir = API_DIR / "model_export"
	store = load_model_store(model_dir)
	app.state.model_store = store
	if store.model_loaded:
		logger.info("Model store ready with %s clusters", store.n_clusters)
	else:
		logger.error("Model store failed to load: %s", store.load_error)
	yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(home.router)
app.include_router(dataset.router)
app.include_router(dashboard.router)
app.include_router(prediction.router)
app.include_router(export.router)

app.include_router(predict_router, prefix="/api/v1", tags=["predict"])