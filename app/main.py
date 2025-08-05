from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.models import (
    ContentRecommendRequest, ContentRecommendResponse, RecommendedContent,
    DataSourceConfig, HealthStatus, ModelMetadata
)
from app.recommender import ContentRecommender, DataSourceAPI
import logging
import os
from typing import Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Content Recommendation API",
    description="고도화된 콘텐츠 추천 서비스 API - API 통신 기반 데이터 소스 지원",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수에서 설정 읽기
def get_data_source_config() -> Optional[DataSourceConfig]:
    """환경 변수에서 데이터 소스 설정 읽기"""
    api_url = os.getenv("CONTENT_API_URL")
    api_key = os.getenv("CONTENT_API_KEY")
    
    if api_url:
        return DataSourceConfig(
            api_base_url=api_url,
            api_key=api_key,
            timeout=int(os.getenv("API_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("API_RETRY_ATTEMPTS", "3")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600"))
        )
    return None

# 추천 시스템 초기화
def get_recommender() -> ContentRecommender:
    """추천 시스템 인스턴스 반환"""
    data_source_config = get_data_source_config()
    
    try:
        recommender = ContentRecommender(
            model_path="app/models/content_sim.pkl",
            data_source_config=data_source_config,
            enable_cache=True
        )
        return recommender
    except Exception as e:
        logger.error(f"추천 시스템 초기화 실패: {e}")
        raise

# 전역 추천 시스템 인스턴스
recommender: Optional[ContentRecommender] = None

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
    global recommender
    try:
        recommender = get_recommender()
        logger.info("추천 시스템이 성공적으로 로드되었습니다.")
    except Exception as e:
        logger.error(f"추천 시스템 로드 실패: {e}")
        recommender = None

@app.get("/")
def read_root():
    """API 상태 확인"""
    return {
        "status": "ok", 
        "service": "Content Recommendation API v2.0",
        "features": [
            "API-based data source support",
            "Caching system",
            "Real-time model refresh",
            "Enhanced filtering and metadata"
        ]
    }

@app.post("/recommend", response_model=ContentRecommendResponse)
def recommend_contents(request: ContentRecommendRequest):
    """콘텐츠 추천 엔드포인트"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="추천 모델이 로드되지 않았습니다.")
    
    try:
        recommendations = recommender.recommend_items(
            content_id=request.content_id,
            top_k=request.top_k,
            use_cache=True
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"Content ID {request.content_id}를 찾을 수 없습니다."
            )
        
        return ContentRecommendResponse(
            query_content_id=request.content_id,
            recommendations=[RecommendedContent(**rec) for rec in recommendations],
            total_count=len(recommendations),
            model_version=recommender.get_model_metadata().version if recommender.get_model_metadata() else None
        )
    
    except Exception as e:
        logger.error(f"추천 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="추천 처리 중 오류가 발생했습니다.")

@app.post("/recommend/refresh", response_model=dict)
def refresh_model(background_tasks: BackgroundTasks):
    """모델 새로고침 엔드포인트"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="추천 모델이 로드되지 않았습니다.")
    
    if not recommender.data_source:
        raise HTTPException(
            status_code=400, 
            detail="API 데이터 소스가 설정되지 않아 모델을 새로고침할 수 없습니다."
        )
    
    # 백그라운드에서 모델 새로고침 실행
    background_tasks.add_task(recommender.refresh_model)
    
    return {
        "message": "모델 새로고침이 백그라운드에서 시작되었습니다.",
        "status": "processing"
    }

@app.get("/health", response_model=HealthStatus)
def health_check():
    """헬스 체크 엔드포인트"""
    if recommender is None:
        return HealthStatus(
            status="unhealthy",
            model_loaded=False,
            data_source_connected=False
        )
    
    health_data = recommender.get_health_status()
    return HealthStatus(**health_data)

@app.get("/model/metadata", response_model=ModelMetadata)
def get_model_metadata():
    """모델 메타데이터 조회"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="추천 모델이 로드되지 않았습니다.")
    
    metadata = recommender.get_model_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="모델 메타데이터를 찾을 수 없습니다.")
    
    return metadata

@app.get("/contents")
def list_contents(limit: int = 10, offset: int = 0):
    """사용 가능한 콘텐츠 목록 조회"""
    if recommender is None or recommender.content_data is None:
        raise HTTPException(status_code=503, detail="추천 모델이 로드되지 않았습니다.")
    
    total = len(recommender.content_data)
    contents = recommender.content_data.iloc[offset:offset+limit]
    
    return {
        "contents": contents.to_dict('records'),
        "total": total,
        "limit": limit,
        "offset": offset
    }

@app.get("/contents/{content_id}")
def get_content(content_id: int):
    """특정 콘텐츠 정보 조회"""
    if recommender is None or recommender.content_data is None:
        raise HTTPException(status_code=503, detail="추천 모델이 로드되지 않았습니다.")
    
    content = recommender.content_data[recommender.content_data['id'] == content_id]
    if content.empty:
        raise HTTPException(status_code=404, detail=f"Content ID {content_id}를 찾을 수 없습니다.")
    
    return content.iloc[0].to_dict()

@app.delete("/cache")
def clear_cache():
    """캐시 초기화"""
    if recommender is None or recommender.cache is None:
        raise HTTPException(status_code=503, detail="캐시 시스템이 사용할 수 없습니다.")
    
    recommender.cache.clear()
    return {"message": "캐시가 초기화되었습니다."}

@app.get("/cache/status")
def get_cache_status():
    """캐시 상태 조회"""
    if recommender is None or recommender.cache is None:
        raise HTTPException(status_code=503, detail="캐시 시스템이 사용할 수 없습니다.")
    
    return recommender.cache.get_status()

# 환경 변수 설정 예시
@app.get("/config/example")
def get_config_example():
    """환경 변수 설정 예시"""
    return {
        "environment_variables": {
            "CONTENT_API_URL": "https://api.example.com/v1",
            "CONTENT_API_KEY": "your-api-key-here",
            "API_TIMEOUT": "30",
            "API_RETRY_ATTEMPTS": "3",
            "CACHE_TTL": "3600"
        },
        "description": "위 환경 변수들을 설정하여 API 기반 데이터 소스를 활성화할 수 있습니다."
    }