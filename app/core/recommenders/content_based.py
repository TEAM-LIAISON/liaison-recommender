# app/core/recommenders/content_based.py
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender
from ...shared.enums import RecommendationType
from ...shared.constants import ALGORITHM_WEIGHTS

class ContentBasedRecommender(BaseRecommender):
    """TF-IDF 기반 콘텐츠 기반 추천"""
    
    def __init__(self, model_loader, spring_boot_client):
        self.model_loader = model_loader
        self.spring_boot_client = spring_boot_client
        self.tfidf_vectorizer = None
        self.content_sim_matrix = None
    
    async def initialize(self):
        """모델 초기화"""
        await self.model_loader.load_all_models()
        self.tfidf_vectorizer = self.model_loader.get_model('tfidf')
        self.content_sim_matrix = self.model_loader.get_model('content_sim')
    
    async def recommend(
        self, 
        user_id: int, 
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """콘텐츠 기반 추천"""
        try:
            # 1. 사용자 구매 이력 조회
            user_purchases = await self.spring_boot_client.get_user_purchases(user_id)
            
            if not user_purchases:
                return []
            
            # 2. 구매한 콘텐츠들의 특성 분석
            user_preferences = self._analyze_user_preferences(user_purchases)
            
            # 3. 유사한 콘텐츠 찾기
            similar_contents = await self._find_similar_contents(
                user_preferences, 
                exclude_ids=[c['content_id'] for c in user_purchases]
            )
            
            return similar_contents[:10]
            
        except Exception as e:
            logger.error(f"콘텐츠 기반 추천 실패: {e}")
            return []
    
    def get_type(self) -> RecommendationType:
        return RecommendationType.CONTENT_BASED
    
    def get_weight(self) -> float:
        return ALGORITHM_WEIGHTS["content_based"]
    
    def _analyze_user_preferences(self, purchases: List[Dict]) -> Dict[str, Any]:
        """사용자 선호도 분석"""
        preferences = {
            'categories': {},
            'content_types': {},
            'price_ranges': {},
            'sellers': {},
            'text_features': []
        }
        
        for purchase in purchases:
            # 카테고리 선호도
            category_id = purchase['category_id']
            preferences['categories'][category_id] = preferences['categories'].get(category_id, 0) + 1
            
            # 콘텐츠 타입 선호도
            content_type = purchase['content_type']
            preferences['content_types'][content_type] = preferences['content_types'].get(content_type, 0) + 1
            
            # 가격대 선호도
            price_range = self._get_price_range(purchase['lowest_price'])
            preferences['price_ranges'][price_range] = preferences['price_ranges'].get(price_range, 0) + 1
            
            # 판매자 선호도
            seller_id = purchase['seller_id']
            preferences['sellers'][seller_id] = preferences['sellers'].get(seller_id, 0) + 1
            
            # 텍스트 특성
            text = f"{purchase['title']} {purchase.get('content_introduction', '')}"
            preferences['text_features'].append(text)
        
        return preferences
    
    async def _find_similar_contents(
        self, 
        preferences: Dict[str, Any], 
        exclude_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """유사한 콘텐츠 찾기"""
        similar_contents = []
        
        # 모든 콘텐츠 조회
        all_contents = await self.spring_boot_client.get_all_contents()
        
        for content in all_contents:
            if content['content_id'] in exclude_ids:
                continue
            
            similarity_score = self._calculate_similarity_score(preferences, content)
            
            if similarity_score > 0.3:
                similar_contents.append({
                    'content_id': content['content_id'],
                    'similarity_score': similarity_score,
                    'title': content['title'],
                    'lowest_price': content['lowest_price'],
                    'content_type': content['content_type'],
                    'category_id': content['category_id'],
                    'seller_id': content['seller_id'],
                    'algorithm': self.get_type().value
                })
        
        return sorted(similar_contents, key=lambda x: x['similarity_score'], reverse=True)
    
    def _calculate_similarity_score(self, preferences: Dict[str, Any], content: Dict) -> float:
        """콘텐츠와 사용자 선호도의 유사도 계산"""
        score = 0.0
        
        # 카테고리 유사도
        if content['category_id'] in preferences['categories']:
            score += 0.3 * (preferences['categories'][content['category_id']] / max(preferences['categories'].values()))
        
        # 콘텐츠 타입 유사도
        if content['content_type'] in preferences['content_types']:
            score += 0.2 * (preferences['content_types'][content['content_type']] / max(preferences['content_types'].values()))
        
        # 가격대 유사도
        price_range = self._get_price_range(content['lowest_price'])
        if price_range in preferences['price_ranges']:
            score += 0.2 * (preferences['price_ranges'][price_range] / max(preferences['price_ranges'].values()))
        
        # 판매자 유사도
        if content['seller_id'] in preferences['sellers']:
            score += 0.1 * (preferences['sellers'][content['seller_id']] / max(preferences['sellers'].values()))
        
        return score
    
    def _get_price_range(self, price: float) -> str:
        """가격대 분류"""
        if price < 10000:
            return "low"
        elif price < 50000:
            return "medium"
        else:
            return "high"