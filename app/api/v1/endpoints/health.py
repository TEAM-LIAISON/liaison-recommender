from fastapi import APIRouter
from datetime import datetime
from app.config import settings

router = APIRouter()

@router.get("/", summary="Health check", tags=["health"])
async def health_check():
    """서비스 상태를 확인하는 헬스 체크 엔드포인트"""
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }