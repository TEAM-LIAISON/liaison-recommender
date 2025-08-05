import pickle
import os
import pandas as pd
import numpy as np
import requests
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from app.models import (
    ContentData, DataSourceConfig, ModelMetadata, 
    ContentType, ContentStatus, RecommendedContent
)


logger = logging.getLogger(__name__)


class DataSourceAPI:
    """API 기반 데이터 소스 클래스"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.api_key}'})
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """API 요청 실행 (재시도 로직 포함)"""
        url = f"{self.config.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.config.retry_attempts}): {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # 지수 백오프
    
    def fetch_contents(self, filters: Optional[Dict] = None) -> List[ContentData]:
        """콘텐츠 데이터 가져오기"""
        params = {'status': 'active'}
        if filters:
            params.update(filters)
        
        try:
            data = self._make_request('/contents', params)
            return [ContentData(**item) for item in data.get('contents', [])]
        except Exception as e:
            logger.error(f"콘텐츠 데이터 가져오기 실패: {e}")
            raise
    
    def fetch_content_by_id(self, content_id: int) -> Optional[ContentData]:
        """특정 콘텐츠 데이터 가져오기"""
        try:
            data = self._make_request(f'/contents/{content_id}')
            return ContentData(**data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"콘텐츠 {content_id} 가져오기 실패: {e}")
            raise


class CacheManager:
    """캐시 관리 클래스"""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 가져오기"""
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """캐시에 값 저장"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        """캐시 초기화"""
        self._cache.clear()
        self._timestamps.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """캐시 상태 반환"""
        return {
            'size': len(self._cache),
            'ttl': self.ttl,
            'keys': list(self._cache.keys())
        }


class ContentRecommender:
    def __init__(self, 
                 model_path: str = "app/models/content_sim.pkl",
                 data_source_config: Optional[DataSourceConfig] = None,
                 enable_cache: bool = True):
        self.model_path = Path(model_path)
        self.data_source_config = data_source_config
        self.enable_cache = enable_cache
        
        # 컴포넌트 초기화
        self.similarity_matrix = None
        self.content_data = None
        self.vectorizer = None
        self.model_metadata = None
        
        # API 데이터 소스 (설정된 경우)
        self.data_source = None
        if data_source_config:
            self.data_source = DataSourceAPI(data_source_config)
        
        # 캐시 매니저
        self.cache = CacheManager() if enable_cache else None
        
        # 모델 로드
        self.load_model()
    
    def load_model(self):
        """학습된 모델과 데이터 로드"""
        if not self.model_path.exists():
            logger.warning(f"모델 파일이 없습니다: {self.model_path}")
            if self.data_source:
                logger.info("API 데이터 소스에서 데이터를 가져와 모델을 초기화합니다.")
                self._initialize_from_api()
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        else:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.similarity_matrix = model_data['similarity_matrix']
                self.content_data = model_data['content_data']
                self.vectorizer = model_data.get('vectorizer')
                self.model_metadata = model_data.get('metadata')
    
    def _initialize_from_api(self):
        """API에서 데이터를 가져와 모델 초기화"""
        if not self.data_source:
            raise ValueError("데이터 소스가 설정되지 않았습니다.")
        
        logger.info("API에서 콘텐츠 데이터를 가져오는 중...")
        contents = self.data_source.fetch_contents()
        
        if not contents:
            raise ValueError("API에서 콘텐츠 데이터를 가져올 수 없습니다.")
        
        # DataFrame으로 변환
        content_dicts = []
        for content in contents:
            content_dicts.append({
                'id': content.id,
                'title': content.title,
                'maker_intro': content.maker_intro,
                'content_introduction': content.content_introduction,
                'content_type': content.content_type.value,
                'tags': ' '.join(content.tags),
                'metadata': json.dumps(content.metadata)
            })
        
        self.content_data = pd.DataFrame(content_dicts)
        self._update_similarity_matrix()
    
    def _update_similarity_matrix(self):
        """유사도 행렬 업데이트"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 텍스트 코퍼스 생성
        corpus = self._create_corpus()
        
        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 메타데이터 업데이트
        self.model_metadata = ModelMetadata(
            version="1.0.0",
            created_at=datetime.now(),
            data_source="api" if self.data_source else "file",
            total_contents=len(self.content_data),
            vectorizer_params=self.vectorizer.get_params(),
            model_size_mb=os.path.getsize(self.model_path) / (1024 * 1024) if self.model_path.exists() else None
        )
    
    def _create_corpus(self) -> List[str]:
        """텍스트 코퍼스 생성"""
        corpus = []
        for _, row in self.content_data.iterrows():
            text_parts = []
            fields = ['title', 'maker_intro', 'content_introduction', 'tags']
            
            for field in fields:
                if field in row and pd.notna(row[field]):
                    text = str(row[field])
                    # HTML 태그 제거
                    text = text.replace('<p>', '').replace('</p>', '')
                    text_parts.append(text.strip())
            
            corpus.append(' '.join(text_parts))
        return corpus
    
    @lru_cache(maxsize=1000)
    def _get_cached_recommendations(self, content_id: int, top_k: int) -> List[Dict[str, Any]]:
        """캐시된 추천 결과 반환"""
        return self._calculate_recommendations(content_id, top_k)
    
    def _calculate_recommendations(self, content_id: int, top_k: int) -> List[Dict[str, Any]]:
        """추천 계산 (캐시 없이)"""
        if content_id not in self.content_data['id'].values:
            return []
        
        idx = self.content_data[self.content_data['id'] == content_id].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        
        recommendations = []
        for i, score in sim_scores:
            content = self.content_data.iloc[i]
            recommendations.append({
                'content_id': int(content['id']),
                'title': content['title'],
                'similarity_score': float(score),
                'maker_intro': content.get('maker_intro'),
                'content_introduction': content.get('content_introduction'),
                'content_type': content.get('content_type'),
                'tags': content.get('tags', '').split() if content.get('tags') else [],
                'metadata': json.loads(content.get('metadata', '{}'))
            })
        
        return recommendations
    
    def recommend_items(self, 
                       content_id: int, 
                       top_k: int = 5,
                       use_cache: bool = True) -> List[Dict[str, Any]]:
        """주어진 content_id에 대해 유사한 콘텐츠 추천"""
        
        # 캐시 사용 여부 확인
        if use_cache and self.cache:
            cache_key = f"rec_{content_id}_{top_k}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"캐시에서 추천 결과 반환: {content_id}")
                return cached_result
        
        # 추천 계산
        recommendations = self._calculate_recommendations(content_id, top_k)
        
        # 캐시에 저장
        if use_cache and self.cache:
            cache_key = f"rec_{content_id}_{top_k}"
            self.cache.set(cache_key, recommendations)
        
        return recommendations
    
    def refresh_model(self):
        """모델 새로고침 (API 데이터 소스가 있는 경우)"""
        if not self.data_source:
            logger.warning("API 데이터 소스가 설정되지 않아 모델을 새로고침할 수 없습니다.")
            return
        
        logger.info("모델 새로고침 시작...")
        self._initialize_from_api()
        
        # 모델 저장
        self.save_model()
        logger.info("모델 새로고침 완료")
    
    def save_model(self):
        """현재 모델 상태 저장"""
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'content_data': self.content_data,
            'vectorizer': self.vectorizer,
            'metadata': self.model_metadata
        }
        
        os.makedirs(self.model_path.parent, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"모델이 저장되었습니다: {self.model_path}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            'status': 'healthy' if self.similarity_matrix is not None else 'unhealthy',
            'model_loaded': self.similarity_matrix is not None,
            'data_source_connected': self.data_source is not None,
            'last_model_update': self.model_metadata.created_at if self.model_metadata else None,
            'cache_status': self.cache.get_status() if self.cache else None,
            'total_contents': len(self.content_data) if self.content_data is not None else 0
        }
    
    def get_model_metadata(self) -> Optional[ModelMetadata]:
        """모델 메타데이터 반환"""
        return self.model_metadata