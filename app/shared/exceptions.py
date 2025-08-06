# app/shared/exceptions.py
from typing import Any, Optional
from fastapi import HTTPException

class RecommendationException(Exception):
    def __init__(
        self,
        message: str,
        error_code: str = "추천 서버 오류",
        http_status: int = 400,
        detail: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.http_status = http_status
        self.detail = detail or message
    
    def to_http_exception(self) -> HTTPException:
        """
        FastAPI HTTPException 으로 변환
        반환 payload:
        {
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail
        }
        """
        raise HTTPException(
            status_code=self.http_status,
            detail={
                "error_code": self.error_code,
                "message": self.message,
                "detail": self.detail,
            },
        )
    
# ◾️ 구체적 예외 클래스들

class ModelLoadException(RecommendationException):
    """추천 모델 로드 실패 예외"""

    def __init__(
        self,
        message: str = "Failed to load recommendation model",
        detail: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            error_code="model_load_error",
            http_status=500,
            detail=detail,
        )


class DataFetchException(RecommendationException):
    """데이터 조회 또는 전처리 실패 예외"""

    def __init__(
        self,
        message: str = "Failed to fetch or preprocess data",
        detail: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            error_code="data_fetch_error",
            http_status=500,
            detail=detail,
        )


class InferenceException(RecommendationException):
    """추천 추론(inference) 중 발생 예외"""

    def __init__(
        self,
        message: str = "Recommendation inference failed",
        detail: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            error_code="inference_error",
            http_status=500,
            detail=detail,
        )


class CacheException(RecommendationException):
    """캐시 처리 중 발생 예외"""

    def __init__(
        self,
        message: str = "Cache operation failed",
        detail: Optional[Any] = None,
    ):
        super().__init__(
            message=message,
            error_code="cache_error",
            http_status=500,
            detail=detail,
        )
