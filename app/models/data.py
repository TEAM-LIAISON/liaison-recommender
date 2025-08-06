from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from .enums import ContentType, ContentStatus


class ContentOption(BaseModel):
    """콘텐츠 옵션 몯레"""
    content_option_id: int
    price: float
    name: str
    option_type: ContentType
    description: Optional[str] = None
    document_file_url: Optional[str] = None
    document_link_url: Optional[str] = None
    document_original_file_name: Optional[str] = None

class ContentData(BaseModel):
    """콘텐츠 데이터 모델"""
    content_id: int
    lowest_price: float
    sale_count: int
    category_id: int
    created_at: datetime
    updated_at: datetime = Field(default_factory=datetime.now)
    user_id: int
    view_count: int
    title: str
    maker_intro: Optional[str] = None
    service_process: Optional[str] = None
    service_target: Optional[str] = None
    content_introduction: Optional[str] = None  # HTML 태그 포함
    content_type: ContentType
    status: ContentStatus = ContentStatus.ACTIVE
    
    # 옵션 정보
    options: List[ContentOption] = []
    
    # 추천 시스템용 메타데이터
    tags: List[str] = []
    metadata: Dict[str, Any] = {}
    
    @property
    def popularity_score(self) -> float:
        """인기도 점수 계산"""
        return (self.sale_count * 10) + (self.view_count * 0.1)
    
    @property
    def price_range(self) -> str:
        """가격대 분류"""
        if self.lowest_price < 10000:
            return "low"
        elif self.lowest_price < 50000:
            return "medium"
        else:
            return "high"

