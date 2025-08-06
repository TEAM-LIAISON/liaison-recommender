# app/dependencies.py
from typing import Generator
from fastapi import Depends
import redis.asyncio as redis

from .config import settings
from .infrastructure.clients.spring_boot_client import SpringBootClient
from .infrastructure.cache.redis_client import RedisClient
from .infrastructure.ml.model_loader import ModelLoader
from .application.services.recommendation_service import RecommendationService
from .application.services.cache_service import CacheService

# Redis 연결
async def get_redis_client() -> Generator[RedisClient, None, None]:
    redis_client = RedisClient(settings.redis_url)
    try:
        yield redis_client
    finally:
        await redis_client.close()

# Spring Boot 클라이언트
async def get_spring_boot_client() -> Generator[SpringBootClient, None, None]:
    client = SpringBootClient(
        base_url=settings.spring_boot_url,
        api_key=settings.spring_boot_api_key,
        timeout=settings.api_timeout,
        retry_attempts=settings.api_retry_attempts
    )
    yield client

# ML 모델 로더
async def get_model_loader() -> Generator[ModelLoader, None, None]:
    loader = ModelLoader(settings.model_path)
    try:
        yield loader
    finally:
        await loader.close()

# 추천 서비스
async def get_recommendation_service(
    redis_client: RedisClient = Depends(get_redis_client),
    spring_boot_client: SpringBootClient = Depends(get_spring_boot_client),
    model_loader: ModelLoader = Depends(get_model_loader)
) -> RecommendationService:
    cache_service = CacheService(redis_client)
    return RecommendationService(
        spring_boot_client=spring_boot_client,
        cache_service=cache_service,
        model_loader=model_loader
    )