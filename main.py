# Sample app: capitalize text and return it

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def capitalize_text(request: Request, user_input: str = Form(...)):
    capitalized_text = user_input.upper() + " 0w0"
    return templates.TemplateResponse("index.html", {"request": request, "capitalized_text": capitalized_text})

app.mount("/static", StaticFiles(directory="static"), name="static")