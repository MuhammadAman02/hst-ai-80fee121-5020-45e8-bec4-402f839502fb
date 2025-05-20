from fastapi import APIRouter

router = APIRouter()

# Import routes to register them with the router
from . import routes