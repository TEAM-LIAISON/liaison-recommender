# app/core/recommenders/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ...shared.enums import RecommendationType
class BaseRecommender(ABC):
    """추천 알고리즘 기본 인터페이스"""

    @abstractmethod
    async def recommend(
        self, 
        user_id: int, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """추천 결과 생성"""
        pass

    @abstractmethod
    def get_type(self) -> RecommendationType:
        """추천 타입 반환"""
        pass
    
    @abstractmethod
    def get_weight(self) -> float:
        """알고리즘 가중치 반환"""
        pass