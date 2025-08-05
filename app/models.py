from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ContentType(str, Enum):
    """콘텐츠 타입 정의"""
    SERVICE = "service"
    PRODUCT = "product"
    ARTICLE = "article"
    VIDEO = "video"


class ContentStatus(str, Enum):
    """콘텐츠 상태 정의"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"


class ContentRecommendRequest(BaseModel):
    content_id: int
    top_k: int = Field(default=5, ge=1, le=20)
    user_id: Optional[int] = None
    content_type: Optional[ContentType] = None
    filters: Optional[Dict[str, Any]] = None


class RecommendedContent(BaseModel):
    content_id: int
    title: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    maker_intro: Optional[str] = None
    content_introduction: Optional[str] = None
    content_type: Optional[ContentType] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ContentRecommendResponse(BaseModel):
    query_content_id: int
    recommendations: List[RecommendedContent]
    total_count: int
    generated_at: datetime = Field(default_factory=datetime.now)
    model_version: Optional[str] = None


class DataSourceConfig(BaseModel):
    """데이터 소스 설정"""
    api_base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    cache_ttl: int = 3600  # 캐시 TTL (초)


class ContentData(BaseModel):
    """콘텐츠 데이터 모델"""
    id: int
    title: str
    maker_intro: Optional[str] = None
    content_introduction: Optional[str] = None
    content_type: ContentType
    status: ContentStatus
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime


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