import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import core components
from .core.config import settings
from .core.logging_config import get_logger
from .core.error_handling import register_exception_handlers

# Initialize main application logger
logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME, # Use setting for title
    description="Enterprise-ready FastAPI application base.",
    version="1.0.0",
    debug=settings.DEBUG, # Use setting for debug mode
    # Add other FastAPI parameters if needed, e.g., lifespan context managers for DB connections
)

# Mount static files directory
# Check both app/static and project root static directories
static_dir_in_app = os.path.join(os.path.dirname(__file__), 'static')
static_dir_in_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')

# First try to use the static directory inside the app directory
if os.path.exists(static_dir_in_app) and os.listdir(static_dir_in_app):
    app.mount("/static", StaticFiles(directory=static_dir_in_app), name="static")
    logger.info(f"Using static directory at {static_dir_in_app}")
# If not found or empty, try the project root static directory
elif os.path.exists(static_dir_in_root) and os.listdir(static_dir_in_root):
    app.mount("/static", StaticFiles(directory=static_dir_in_root), name="static")
    logger.info(f"Using static directory at {static_dir_in_root}")
else:
    logger.warning(f"No static files found in {static_dir_in_app} or {static_dir_in_root}")

# Configure Jinja2 templates
# Check both app/templates and project root templates directories
templates_dir_in_app = os.path.join(os.path.dirname(__file__), 'templates')
templates_dir_in_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

# First try to use the templates directory inside the app directory
if os.path.exists(templates_dir_in_app) and os.listdir(templates_dir_in_app):
    templates = Jinja2Templates(directory=templates_dir_in_app)
    logger.info(f"Using templates directory at {templates_dir_in_app}")
# If not found or empty, try the project root templates directory
elif os.path.exists(templates_dir_in_root) and os.listdir(templates_dir_in_root):
    templates = Jinja2Templates(directory=templates_dir_in_root)
    logger.info(f"Using templates directory at {templates_dir_in_root}")
else:
    templates = None
    logger.warning(f"No templates found in {templates_dir_in_app} or {templates_dir_in_root}")

# Import and include routers after app creation
from .api import routes as api_routes
from .frontend import routes as frontend_routes

# Include routers
app.include_router(api_routes.router, prefix="/api", tags=["api"])
app.include_router(frontend_routes.router, tags=["frontend"])

# Note: The application is designed to be extensible.
# When AI-generated code is added, it can be placed in the 'generated' directory
# and imported here with its own router.

# Register custom exception handlers
register_exception_handlers(app)

# Add root endpoint (optional)
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the FastAPI application!"}

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} ({settings.APP_ENV})")
    # Add any startup tasks here (database connections, etc.)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.APP_NAME}")
    # Add any cleanup tasks here