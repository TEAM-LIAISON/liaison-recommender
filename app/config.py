import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """콘텐츠 추천 서버 설정"""

    # FastAPI 설정
    app_name: str = "Groble Content Recommendation API"
    app_version: str = "1.0.0"
    debug: bool = False

    spring_boot_url: str = os.getenv("SPRING_BOOT_URL", "http://localhost:8080")
    api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    api_retry_attempts: int = int(os.getenv("API_RETRY_ATTEMPTS", "5"))

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