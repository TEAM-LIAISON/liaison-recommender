from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .enums import ContentType


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