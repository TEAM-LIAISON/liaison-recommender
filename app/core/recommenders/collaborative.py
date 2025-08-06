# app/core/recommenders/collaborative.py
import logging
import numpy as np
from typing import List, Dict, Any
from .base import BaseRecommender
from ...shared.enums import RecommendationType
from ...shared.constants import ALGORITHM_WEIGHTS

logger = logging.getLogger(__name__)

class CollaborativeRecommender(BaseRecommender):
    """협업 필터링 추천"""

    def __init__(self, model_loader, spring_boot_client):
        self.model_loader = model_loader
        self.spring_boot_client = spring_boot_client
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None

    async def initialize(self):
        """모델 초기화"""
        await self.model_loader.load_all_models()
        # 모델로부터 사용자-아이템 행렬, ID 리스트 로드
        self.user_ids, self.item_ids, self.user_item_matrix = \
            self.model_loader.get_model('user_item_matrix')

    async def recommend(
        self,
        user_id: int,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """협업 필터링을 통한 추천"""
        try:
            if user_id not in self.user_ids:
                return []
            idx = self.user_ids.index(user_id)
            # 코사인 유사도 기반으로 유사 사용자 찾기
            sim = np.dot(self.user_item_matrix, self.user_item_matrix[idx])
            sim_scores = sim / (np.linalg.norm(self.user_item_matrix, axis=1) * np.linalg.norm(self.user_item_matrix[idx]) + 1e-8)
            # 유사도 상위 사용자들의 아이템 평점 합산
            weighted = np.dot(sim_scores, self.user_item_matrix)
            # 이미 소비한 아이템 제외
            user_vector = self.user_item_matrix[idx]
            weighted[user_vector > 0] = 0
            # Top items
            top_indices = np.argsort(weighted)[::-1]
            recs = []
            for i in top_indices:
                score = weighted[i]
                if score <= 0:
                    break
                recs.append({'content_id': int(self.item_ids[i]), 'score': float(score), 'algorithm': self.get_type().value})
            return recs
        except Exception as e:
            logger.error(f"협업 필터링 추천 실패: {e}")
            return []

    def get_type(self) -> RecommendationType:
        return RecommendationType.COLLABORATIVE

    def get_weight(self) -> float:
        return ALGORITHM_WEIGHTS["collaborative"]