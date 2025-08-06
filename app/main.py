# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .config import settings
from .api.v1.endpoints.recommendations import router as rec_router
from .api.v1.endpoints.health import router as health_router

from .infrastructure.ml.model_loader import ModelLoader
from .application.batch.scheduler import RecommendationScheduler

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(
    rec_router,
    prefix="/api/v1/recommendations",
    tags=["recommendations"]
)
app.include_router(
    health_router,
    prefix="/api/v1/health",
    tags=["health"]
)

# 전역 스케줄러 객체
scheduler: RecommendationScheduler | None = None

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
    global scheduler
    try:
        # 모델 로더 생성 후 스케줄러 초기화
        model_loader = ModelLoader(settings)
        scheduler = RecommendationScheduler(model_loader)

        logger.info("추천 시스템이 성공적으로 시작되었습니다.")
    except Exception:
        logger.exception("시스템 시작 중 오류 발생")

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 정리"""
    global scheduler
    if scheduler:
        # APScheduler 자체 스케줄러 종료
        scheduler.scheduler.shutdown(wait=False)
        logger.info("추천 시스템이 종료되었습니다.")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Groble Recommendation API",
        "version": settings.app_version,
        "status": "running"
    }
