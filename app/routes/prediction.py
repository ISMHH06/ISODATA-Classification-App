from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/prediction")
def prediction_page(request: Request):
    return templates.TemplateResponse(
        "prediction.html",
        {"request": request}
    )