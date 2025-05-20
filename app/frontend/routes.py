from fastapi import Request
from fastapi.responses import HTMLResponse
from . import router
from .. import templates
import os
from datetime import datetime

@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main index page using Jinja2 templates."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not templates:
        error_msg = "Templates directory not configured or not found. Please run setup_deployment.py before deployment."
        logger.error(error_msg)
        return HTMLResponse(
            content=f"<html><body><h1>Configuration Error</h1><p>{error_msg}</p></body></html>",
            status_code=500
        )
    
    try:
        # Check if we can access the template directory
        if not templates.env.loader.searchpath:
            error_msg = "Template loader has no searchpaths configured"
            logger.error(error_msg)
            return HTMLResponse(
                content=f"<html><body><h1>Template Error</h1><p>{error_msg}</p></body></html>",
                status_code=500
            )
            
        template_path = os.path.join(templates.env.loader.searchpath[0], 'index.html')
        logger.info(f"Looking for template at: {template_path}")
        
        if os.path.exists(template_path):
            # Add current_time to the template context
            return templates.TemplateResponse("index.html", {
                "request": request,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_year": datetime.now().year
            })
        else:
            error_msg = f"Frontend template 'index.html' not found at {template_path}"
            logger.error(error_msg)
            return HTMLResponse(
                content=f"<html><body><h1>Template Not Found</h1><p>{error_msg}</p></body></html>",
                status_code=404
            )
    except Exception as e:
        error_msg = f"Error rendering template: {str(e)}"
        logger.exception(error_msg)
        return HTMLResponse(
            content=f"<html><body><h1>Template Error</h1><p>{error_msg}</p></body></html>",
            status_code=500
        )

# Add additional frontend routes here using the @router decorator