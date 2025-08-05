from pydantic import BaseModel
from typing import Optional


class DataSourceConfig(BaseModel):
    """데이터 소스 설정"""
    api_base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    cache_ttl: int = 3600  # 캐시 TTL (초) 