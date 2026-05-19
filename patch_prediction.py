import pandas as pd
from app.utils.dataset_store import UPLOAD_DIR, load_dataset_summary
from fastapi import APIRouter
from app.routes.prediction import router

@router.get("/api/dataset/sample")
def sample_record():
    summary = load_dataset_summary()
    if not summary or "dataset" not in summary:
        return {"error": "No dataset uploaded"}
    
    path = UPLOAD_DIR / summary["dataset"]
    if not path.exists():
        return {"error": "Dataset file not found"}
        
    df = pd.read_csv(path)
    sample = df.sample(1).iloc[0].to_dict()
    return {"sample": sample}
