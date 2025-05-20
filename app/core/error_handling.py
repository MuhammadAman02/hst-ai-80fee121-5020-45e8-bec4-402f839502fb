from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging_config import get_logger

logger = get_logger(__name__)

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTPException: {exc.status_code} {exc.detail} for {request.method} {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_messages.append(f"Field '{field}': {message}")
    
    logger.warning(f"RequestValidationError: {error_messages} for {request.method} {request.url.path} - Body: {await request.body()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )

async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Pydantic ValidationError: {exc.errors()} for {request.method} {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Pydantic Validation Error", "errors": exc.errors()},
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Unhandled exception: {exc} for {request.method} {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )

def register_exception_handlers(app):
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    logger.info("Custom exception handlers registered.")

# The registration function should be called in the main application setup (app/__init__.py)