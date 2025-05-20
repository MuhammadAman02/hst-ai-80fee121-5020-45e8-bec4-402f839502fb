from fastapi import APIRouter

# Create router
router = APIRouter()

@router.get('/ping')
async def ping_pong():
    """A simple ping endpoint."""
    return {"message": "pong!"}

# Add additional API routes here using the @router decorator
