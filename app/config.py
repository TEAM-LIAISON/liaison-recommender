# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pydantic_settings import SettingsConfigDict

class Settings(BaseSettings):
    """애플리케이션 설정"""

    # Pydantic v2: 환경변수 로딩 및 추가 필드 무시
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = Field("Groble Recommendation API", env="APP_NAME")
    app_version: str = Field("0.1.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    model_dir: str = Field("./models", env="MODEL_DIR")
    server_timezone: str = Field("Asia/Seoul", env="SERVER_TIMEZONE")

# 설정 인스턴스
settings = Settings()