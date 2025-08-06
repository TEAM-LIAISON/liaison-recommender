# app/schemas/response.py
from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationResponse(BaseModel):
    """Recommendation API 응답 스키마"""

    recommendations: List[int] = Field(
        ..., description="추천된 콘텐츠 ID 리스트"
    )
    count: int = Field(
        ..., description="추천 결과 항목 개수"
    )
    algorithm: Optional[str] = Field(
        None, description="사용된 추천 알고리즘 타입"
    )

    class Config:
        orm_mode = True

    @classmethod
    def from_list(cls, rec_list: List[int], algorithm: Optional[str] = None):
        """
        추천 ID 리스트와 알고리즘 이름으로 RecommendationResponse 생성
        """
        return cls(
            recommendations=rec_list,
            count=len(rec_list),
            algorithm=algorithm
        )
