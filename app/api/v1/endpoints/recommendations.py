# app/api/v1/endpoints/recommendations.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from ..schemas.request import RecommendationRequest
from ..schemas.response import RecommendationResponse
from ....application.services.recommendation_service import RecommendationService

router = APIRouter()

@router.post("/recommend", response_model=List[RecommendationResponse])
async def get_recommendations(
    request: RecommendationRequest,
    recommendation_service: RecommendationService = Depends()
):
    """추천 결과 조회"""
    try:
        recommendations = await recommendation_service.get_recommendations(
            user_id=request.user_id,
            context=request.context,
            use_cache=request.use_cache
        )
        
        return [
            RecommendationResponse(
                content_id=rec['content_id'],
                title=rec['title'],
                lowest_price=rec['lowest_price'],
                content_type=rec['content_type'],
                final_score=rec['final_score'],
                algorithms=rec['algorithms']
            )
            for rec in recommendations
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/batch/{user_id}")
async def get_batch_recommendations(
    user_id: int,
    limit: int = 10,
    recommendation_service: RecommendationService = Depends()
):
    """배치 추천 결과 조회 (Spring Boot용)"""
    try:
        recommendations = await recommendation_service.get_recommendations(
            user_id=user_id,
            use_cache=True
        )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations[:limit],
            "total_count": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))