import csv
import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/export")
def export_page(request: Request):
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    context = {
        "request": request,
        "metadata": metadata,
    }
    return templates.TemplateResponse(name="export.html", context=context)


@router.get("/export/metadata.json")
def export_metadata(request: Request) -> JSONResponse:
    store = getattr(request.app.state, "model_store", None)
    metadata = store.metadata if store and store.metadata else {}
    return JSONResponse(content=metadata)


@router.get("/export/cluster-profiles.csv")
def export_cluster_profiles(request: Request) -> StreamingResponse:
    store = getattr(request.app.state, "model_store", None)
    profiles = store.cluster_profiles if store and store.cluster_profiles else {}
    metadata = store.metadata if store and store.metadata else {}
    feature_columns = metadata.get("feature_columns") or []

    if not feature_columns and profiles:
        sample_key = next(iter(profiles.keys()))
        feature_columns = list(profiles[sample_key].keys())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["cluster_id"] + feature_columns)

    for cluster_key in sorted(profiles.keys(), key=lambda value: int(value)):
        row = [cluster_key]
        profile = profiles[cluster_key]
        for column in feature_columns:
            row.append(profile.get(column, ""))
        writer.writerow(row)

    output.seek(0)
    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=cluster_profiles.csv"
    return response


@router.get("/export/artifacts.zip")
def export_artifacts() -> StreamingResponse:
    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "isodata_api" / "model_export"
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in model_dir.glob("*"):
            if path.is_file():
                archive.write(path, arcname=path.name)

    buffer.seek(0)
    response = StreamingResponse(buffer, media_type="application/zip")
    response.headers["Content-Disposition"] = "attachment; filename=model_export.zip"
    return response