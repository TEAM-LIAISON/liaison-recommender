from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class ModelMetadata(BaseModel):
    """모델 메타데이터"""
    version: str
    created_at: datetime
    data_source: str
    total_contents: int
    vectorizer_params: Dict[str, Any]
    similarity_algorithm: str = "cosine"
    model_size_mb: Optional[float] = None


class HealthStatus(BaseModel):
    """시스템 상태"""
    status: str
    model_loaded: bool
    data_source_connected: bool
    last_model_update: Optional[datetime] = None
    cache_status: Optional[Dict[str, Any]] = None 