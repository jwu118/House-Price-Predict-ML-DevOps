
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from controllers.controller import router


# Get the directory containing this main.py file.
SERVER_DIR = Path(__file__).resolve().parent

# Find the templates and static folders.
TEMPLATE_DIR = SERVER_DIR / "templates"
STATIC_DIR = SERVER_DIR / "static"

# Create the FastAPI application.
app = FastAPI()

# Register the prediction endpoint from controller.py.
app.include_router(router)

# Tell FastAPI where index.html is located.
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Serve CSS, JavaScript, and other static files if the folder exists.
if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


# Display index.html when someone visits the homepage.
@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# Start the server when running this file directly.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
