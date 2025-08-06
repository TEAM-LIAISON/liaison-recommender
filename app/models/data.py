from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

from .enums import ContentType, ContentStatus


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
class ContentOptoin(BaseModel):
    """콘텐츠 옵션 몯레"""
    content_option_id: int
    price: float
    name: str
    option_type: ContentType
    description: Optional[str] = None
    document_file_url: Optional[str] = None
    document_link_url: Optional[str] = None
    document_original_file_name: Optional[str] = None
