from pathlib import Path
import pickle
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ModelLoader:
    """추천 모델 로드 및 캐싱 유틸리티"""

    def __init__(self, settings):
        # settings.model_dir 또는 settings.MODEL_DIR 등 경로 설정
        model_dir = getattr(settings, 'model_dir', None)
        if model_dir is None:
            raise ValueError("settings.model_dir가 정의되어 있지 않습니다.")
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise ValueError(f"모델 디렉토리를 찾을 수 없습니다: {self.model_dir}")
        self._models: Dict[str, Any] = {}

    async def load_all_models(self) -> None:
        """모든 모델 파일을 읽어서 메모리에 로드"""
        # 예시: tfidf.pkl, content_sim.pkl, user_item_matrix.pkl, item_embeddings.pkl, kmeans.pkl
        for model_name in ['tfidf', 'content_sim', 'user_item_matrix', 'item_embeddings', 'kmeans']:
            path = self.model_dir / f"{model_name}.pkl"
            try:
                with path.open('rb') as f:
                    self._models[model_name] = pickle.load(f)
                logger.info(f"Loaded model: {model_name} from {path}")
            except FileNotFoundError:
                logger.warning(f"모델 파일 누락: {path}")
            except Exception as e:
                logger.error(f"모델 로드 중 오류: {model_name}, {e}")

    def get_model(self, name: str) -> Any:
        """이전에 로드된 모델 반환"""
        model = self._models.get(name)
        if model is None:
            raise ValueError(f"모델이 로드되지 않았습니다: {name}")
        return model
