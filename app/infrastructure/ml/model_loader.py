import joblib
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from ...shared.exceptions import ModelLoadException

logger = logging.getLogger(__name__)

class ModelLoader:
    """ML 모델 로더"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.models = {}
        self.metadata = {}
    
    async def load_all_models(self):
        """모든 모델 로드"""
        try:
            # TF-IDF 벡터라이저
            tfidf_path = self.model_path / "tfidf_vectorizer.joblib"
            if tfidf_path.exists():
                self.models['tfidf'] = joblib.load(tfidf_path)
                logger.info("TF-IDF 벡터라이저 로드 완료")
            
            # SVD 모델
            svd_path = self.model_path / "svd_model.joblib"
            if svd_path.exists():
                self.models['svd'] = joblib.load(svd_path)
                logger.info("SVD 모델 로드 완료")
            
            # K-means 모델
            kmeans_path = self.model_path / "kmeans_model.joblib"
            if kmeans_path.exists():
                self.models['kmeans'] = joblib.load(kmeans_path)
                logger.info("K-means 모델 로드 완료")
            
            # 콘텐츠 유사도 매트릭스
            sim_path = self.model_path / "content_sim.pkl"
            if sim_path.exists():
                with open(sim_path, 'rb') as f:
                    self.models['content_sim'] = pickle.load(f)
                logger.info("콘텐츠 유사도 매트릭스 로드 완료")
            
            # 모델 메타데이터
            metadata_path = self.model_path / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("모델 메타데이터 로드 완료")
            
            logger.info(f"총 {len(self.models)}개 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise ModelLoadException(f"모델 로드 실패: {e}")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """특정 모델 조회"""
        return self.models.get(model_name)
    
    def get_metadata(self) -> Dict[str, Any]:
        """모델 메타데이터 조회"""
        return self.metadata
    
    async def close(self):
        """리소스 정리"""
        self.models.clear()
        self.metadata.clear()