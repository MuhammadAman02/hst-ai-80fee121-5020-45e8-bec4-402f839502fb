import logging
import sys
from logging.handlers import RotatingFileHandler

# Define log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create a custom logger
logger = logging.getLogger("app")
logger.setLevel(logging.INFO) # Default level, can be overridden by config

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

# File handler (optional, but good for production)
# Creates a logs directory if it doesn't exist
import os
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=1024*1024*5, backupCount=5, encoding='utf-8') # 5MB per file, 5 backup files
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name, inheriting base config."""
    return logging.getLogger(name)

# End of logging configuration