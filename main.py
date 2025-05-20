import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Determine which framework to use based on environment variable
# Default to FastAPI if not specified
FRAMEWORK = os.getenv("FRAMEWORK", "fastapi").lower()

# Import the appropriate application based on the framework
if FRAMEWORK == "nicegui":
    try:
        from nicegui import ui, app as nicegui_app
        # Setup NiceGUI app here
        application = nicegui_app
    except ImportError:
        print("NiceGUI not installed. Please install with: pip install nicegui")
        exit(1)
else:
    # Default to FastAPI
    from app import app
    application = app

# This is used by ASGI servers like Uvicorn
app = application

if __name__ == "__main__":
    import uvicorn
    # Run the application with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)