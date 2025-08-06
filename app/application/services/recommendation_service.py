# app/application/services/recommendation_service.py
import logging
from typing import List, Dict, Any, Optional
from ...core.recommenders.base import BaseRecommender
from ...core.recommenders.content_based import ContentBasedRecommender
from ...core.recommenders.collaborative import CollaborativeRecommender
from ...core.recommenders.clustering import ClusteringRecommender
from ...core.recommenders.hybrid import HybridRecommender
from ...shared.exceptions import RecommendationException

logger = logging.getLogger(__name__)

class RecommendationService:
    """추천 서비스"""
    
    def __init__(self, spring_boot_client, cache_service, model_loader):
        self.spring_boot_client = spring_boot_client
        self.cache_service = cache_service
        self.model_loader = model_loader
        
        # 추천 알고리즘 인스턴스 생성
        self.recommenders = [
            ContentBasedRecommender(model_loader, spring_boot_client),
            CollaborativeRecommender(model_loader, spring_boot_client),
            ClusteringRecommender(model_loader, spring_boot_client),
            HybridRecommender(model_loader, spring_boot_client),
        ]

    async def initialize_recommenders(self):
        """비동기 추천 알고리즘 초기화"""
        for recommender in self.recommenders:
            init = getattr(recommender, 'initialize', None)
            if callable(init):
                # recommender.initialize는 async def 여야 합니다
                await init()
    
    async def get_recommendations(
        self, 
        user_id: int, 
        context: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """통합 추천 결과 생성"""
        try:
            # 1. 캐시 확인
            if use_cache:
                cache_key = f"recommendations:{user_id}"
                cached_result = await self.cache_service.get(cache_key)
                if cached_result:
                    return cached_result
            
            # 2. 각 추천 알고리즘 실행
            all_recommendations = []
            for recommender in self.recommenders:
                try:
                    recommendations = await recommender.recommend(user_id, context)
                    all_recommendations.extend(recommendations)
                except Exception as e:
                    logger.error(f"{recommender.get_type().value} 추천 실패: {e}")
                    continue
            
            # 3. 결과 집계 및 정렬
            final_recommendations = self._aggregate_recommendations(all_recommendations)
            
            # 4. 캐시 저장
            if use_cache:
                await self.cache_service.set(cache_key, final_recommendations)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"추천 생성 실패: {e}")
            raise RecommendationException(f"추천 생성 실패: {e}")
    
    def _aggregate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """추천 결과 집계"""
        # 콘텐츠별로 점수 집계
        content_scores = {}
        
        for rec in recommendations:
            content_id = rec['content_id']
            algorithm_weight = rec.get('algorithm_weight', 1.0)
            similarity_score = rec['similarity_score']
            
            if content_id not in content_scores:
                content_scores[content_id] = {
                    'content_id': content_id,
                    'title': rec['title'],
                    'lowest_price': rec['lowest_price'],
                    'content_type': rec['content_type'],
                    'category_id': rec['category_id'],
                    'seller_id': rec['seller_id'],
                    'final_score': 0.0,
                    'algorithms': []
                }
            
            content_scores[content_id]['final_score'] += similarity_score * algorithm_weight
            content_scores[content_id]['algorithms'].append(rec['algorithm'])
        
        # 최종 점수로 정렬
        final_recommendations = list(content_scores.values())
        final_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_recommendations[:10]