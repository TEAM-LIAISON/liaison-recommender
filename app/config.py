# app/config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings  # 변경된 부분

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # FastAPI 설정
    app_name: str = "Groble Recommendation API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Spring Boot API 설정
    spring_boot_url: str = os.getenv("SPRING_BOOT_URL", "http://localhost:8080")
    spring_boot_api_key: Optional[str] = os.getenv("SPRING_BOOT_API_KEY")
    api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    api_retry_attempts: int = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
    
    # Redis 설정
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_ttl: int = int(os.getenv("REDIS_TTL", "3600"))
    
    # ML 모델 설정
    model_path: str = os.getenv("MODEL_PATH", "app/infrastructure/models")
    
    # 배치 설정
    batch_update_interval: int = int(os.getenv("BATCH_UPDATE_INTERVAL", "3600"))
    
    class Config:
        env_file = ".env"

settings = Settings()