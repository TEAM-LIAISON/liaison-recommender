from pydantic import BaseModel, Field, root_validator
from typing import Optional
from app.shared.enums import RecommendationType


class RecommendationRequest(BaseModel):
    """Recommendation API 요청 스키마"""

    user_id: Optional[int] = Field(
        None, description="추천 대상 사용자 ID (협업 필터링용)")
    content_id: Optional[int] = Field(
        None, description="추천 기준 콘텐츠 ID (콘텐츠 기반 필터링용)")
    type: RecommendationType = Field(
        ..., description="사용할 추천 알고리즘 타입")
    top_k: int = Field(
        10, gt=0, description="반환할 추천 결과 개수 (1 이상의 정수)")

    def validate_identifiers(cls, values):
        user_id = values.get('user_id')
        content_id = values.get('content_id')
        if not (user_id or content_id):
            raise ValueError("user_id 또는 content_id 중 하나는 반드시 제공되어야 합니다.")
        if user_id and content_id:
            raise ValueError("user_id와 content_id는 동시에 설정할 수 없습니다.")
        return values