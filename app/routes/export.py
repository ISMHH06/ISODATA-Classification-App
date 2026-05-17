from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/export")
def export_page(request: Request):
    return templates.TemplateResponse(
        "export.html",
        {"request": request}
    )