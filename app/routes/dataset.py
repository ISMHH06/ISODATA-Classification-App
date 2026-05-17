from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/dataset")
def dataset_page(request: Request):
    return templates.TemplateResponse(
        "dataset.html",
        {"request": request}
    )