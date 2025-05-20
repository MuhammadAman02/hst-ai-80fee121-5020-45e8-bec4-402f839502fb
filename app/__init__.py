from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from skimage import color
from PIL import Image
import io
import base64
import os

app = FastAPI()

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configure Jinja2 templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Extract the L channel
    l_channel = lab_image[:,:,0]
    
    # Calculate average skin tone
    avg_skin_tone = np.mean(l_channel)
    
    # Determine skin tone category
    if avg_skin_tone < 50:
        skin_tone = "Dark"
    elif avg_skin_tone < 120:
        skin_tone = "Medium"
    else:
        skin_tone = "Light"
    
    # Suggest color combinations based on skin tone
    color_combinations = get_color_combinations(skin_tone)
    
    # Convert image to base64 for display
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "skin_tone": skin_tone,
        "color_combinations": color_combinations,
        "image": img_str
    }

@app.post("/change_skin_tone")
async def change_skin_tone(file: UploadFile = File(...), new_tone: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Adjust L channel based on new_tone
    l_channel = lab_image[:,:,0]
    if new_tone == "darker":
        l_channel = np.clip(l_channel - 20, 0, 255)
    elif new_tone == "lighter":
        l_channel = np.clip(l_channel + 20, 0, 255)
    
    lab_image[:,:,0] = l_channel
    
    # Convert back to BGR
    modified_img = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    
    # Convert image to base64 for display
    img_pil = Image.fromarray(cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": img_str}

def get_color_combinations(skin_tone):
    combinations = {
        "Dark": ["Gold", "Orange", "Red", "Pink", "Purple"],
        "Medium": ["Blue", "Green", "Purple", "Red", "Orange"],
        "Light": ["Navy", "Burgundy", "Forest Green", "Lavender", "Pastel Pink"]
    }
    return combinations.get(skin_tone, [])

# Include routers
from .api import routes as api_routes
from .frontend import routes as frontend_routes

app.include_router(api_routes.router, prefix="/api", tags=["api"])
app.include_router(frontend_routes.router, tags=["frontend"])