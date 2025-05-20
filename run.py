import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    reload = os.getenv("APP_RELOAD", "true").lower() == "true"

    # Run the FastAPI app using Uvicorn
    # 'app:app' refers to the 'app' instance in the 'app' module (app/__init__.py)
    uvicorn.run("app:app", host=host, port=port, reload=reload)