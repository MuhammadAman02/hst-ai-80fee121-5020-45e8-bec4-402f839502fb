from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "My Enterprise App"
    APP_VERSION: str = "1.0.0"  # Semantic versioning
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = False
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./default.db")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key")
    
    # Optional API keys and external service configurations
    openai_api_key: Optional[str] = None
    langchain_api_key: Optional[str] = None
    langchain_endpoint: Optional[str] = None
    langchain_project: Optional[str] = None
    langchain_tracing_v2: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None
    github_token: Optional[str] = None
    github_username: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in environment variables

settings = Settings()