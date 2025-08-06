# app/core/recommenders/hybrid.py
from typing import List, Dict, Any
import logging
import numpy as np
from .base import BaseRecommender
from ...shared.enums import RecommendationType
from ...shared.constants import ALGORITHM_WEIGHTS

logger = logging.getLogger(__name__)

import app.core.recommenders.content_based as ContentBasedRecommender
import app.core.recommenders.collaborative as CollaborativeRecommender

class HybridRecommender(BaseRecommender):
    """하이브리드 추천: 가중 합산"""

    def __init__(self, model_loader, spring_boot_client):
        self.model_loader = model_loader
        self.spring_boot_client = spring_boot_client
        # 내부 리코멘더들
        self.content_rec = ContentBasedRecommender(model_loader, spring_boot_client)
        self.collab_rec = CollaborativeRecommender(model_loader, spring_boot_client)

    async def initialize(self):
        await self.content_rec.initialize()
        await self.collab_rec.initialize()

    async def recommend(
        self,
        user_id: int,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """콘텐츠 + 협업 필터링 가중 합산"""
        # 각각 추천 리스트 가져오기
        cb = await self.content_rec.recommend(user_id, context)
        cf = await self.collab_rec.recommend(user_id, context)
        # content id -> score 매핑
        scores = {}
        for rec in cb:
            scores[rec['content_id']] = scores.get(rec['content_id'], 0) + rec['similarity_score'] * self.content_rec.get_weight()
        for rec in cf:
            scores[rec['content_id']] = scores.get(rec['content_id'], 0) + rec['score'] * self.collab_rec.get_weight()
        # 최종 정렬
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{'content_id': cid, 'score': float(score), 'algorithm': self.get_type().value} for cid, score in sorted_ids]

    def get_type(self) -> RecommendationType:
        return RecommendationType.HYBRID

    def get_weight(self) -> float:
        return ALGORITHM_WEIGHTS["hybrid"]
