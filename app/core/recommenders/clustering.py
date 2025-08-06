
# app/core/recommenders/clustering.py
import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseRecommender
from ...shared.enums import RecommendationType
from ...shared.constants import ALGORITHM_WEIGHTS

logger = logging.getLogger(__name__)

class ClusteringRecommender(BaseRecommender):
    """KMeans 클러스터링 기반 추천"""

    def __init__(self, model_loader, spring_boot_client, n_clusters: int = 10):
        self.model_loader = model_loader
        self.spring_boot_client = spring_boot_client
        self.n_clusters = n_clusters
        self.cluster_model: KMeans = None
        self.embeddings = None
        self.item_ids = None

    async def initialize(self):
        """모델 초기화"""
        await self.model_loader.load_all_models()
        self.cluster_model = self.model_loader.get_model('kmeans')
        # embeddings: numpy array, item_ids: list
        self.item_ids, self.embeddings = self.model_loader.get_model('item_embeddings')

    async def recommend(
        self,
        user_id: int,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """클러스터링 기반 추천"""
        try:
            # 추천 기준 벡터
            if context and 'vector' in context:
                vec = np.array(context['vector'])
            else:
                # 기본: 사용자 기본 행동 기반 벡터 요청
                vec = await self.spring_boot_client.get_user_vector(user_id)
            cluster = self.cluster_model.predict([vec])[0]
            members = np.where(self.cluster_model.labels_ == cluster)[0]
            recs = []
            for idx in members:
                recs.append({'content_id': int(self.item_ids[idx]), 'algorithm': self.get_type().value})
            return recs
        except Exception as e:
            logger.error(f"클러스터링 추천 실패: {e}")
            return []

    def get_type(self) -> RecommendationType:
        return RecommendationType.CLUSTERING

    def get_weight(self) -> float:
        return ALGORITHM_WEIGHTS["clustering"]