#!/usr/bin/env python

import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_deployment():
    """Set up the deployment by ensuring templates and static files are in the correct locations."""
    # Get the project base directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define source and destination directories
    templates_src = os.path.join(project_dir, 'templates')
    templates_dest = os.path.join(project_dir, 'app', 'templates')
    
    static_src = os.path.join(project_dir, 'static')
    static_dest = os.path.join(project_dir, 'app', 'static')
    
    # Create destination directories if they don't exist
    os.makedirs(templates_dest, exist_ok=True)
    os.makedirs(static_dest, exist_ok=True)
    
    # Copy templates
    if os.path.exists(templates_src) and os.listdir(templates_src):
        logger.info(f"Copying templates from {templates_src} to {templates_dest}")
        for item in os.listdir(templates_src):
            src_item = os.path.join(templates_src, item)
            dest_item = os.path.join(templates_dest, item)
            
            if os.path.isdir(src_item):
                if os.path.exists(dest_item):
                    shutil.rmtree(dest_item)
                shutil.copytree(src_item, dest_item)
            else:
                shutil.copy2(src_item, dest_item)
    else:
        logger.warning(f"No templates found in {templates_src}")
    
    # Copy static files
    if os.path.exists(static_src) and os.listdir(static_src):
        logger.info(f"Copying static files from {static_src} to {static_dest}")
        for item in os.listdir(static_src):
            src_item = os.path.join(static_src, item)
            dest_item = os.path.join(static_dest, item)
            
            if os.path.isdir(src_item):
                if os.path.exists(dest_item):
                    shutil.rmtree(dest_item)
                shutil.copytree(src_item, dest_item)
            else:
                shutil.copy2(src_item, dest_item)
    else:
        logger.warning(f"No static files found in {static_src}")
    
    # Verify the setup
    if os.path.exists(templates_dest) and os.listdir(templates_dest):
        logger.info(f"Templates successfully set up in {templates_dest}")
    else:
        logger.error(f"Failed to set up templates in {templates_dest}")
    
    if os.path.exists(static_dest) and os.listdir(static_dest):
        logger.info(f"Static files successfully set up in {static_dest}")
    else:
        logger.error(f"Failed to set up static files in {static_dest}")

if __name__ == "__main__":
    logger.info("Starting deployment setup...")
    setup_deployment()
    logger.info("Deployment setup completed.")