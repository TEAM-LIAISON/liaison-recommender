# app/infrastructure/cache/redis_client.py
import redis.asyncio as redis
import json
import logging
from typing import Optional, Any, Dict
from ...shared.exceptions import CacheException

logger = logging.getLogger(__name__)
class RedisClient:
    """Redis 캐시 클라이언트"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None

    async def connect(self):
        """Redis 연결"""
        try:
            self.client = redis.from_url(self.redis_url)
            await self.client.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise CacheException(f"Redis 연결 실패: {e}")
    
    async def close(self):
        """Redis 연결 종료"""
        if self.client:
            await self.client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        try:
            if not self.client:
                await self.connect()
            
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"캐시 조회 실패: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """캐시에 값 저장"""
        try:
            if not self.client:
                await self.connect()
            
            await self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """캐시에서 값 삭제"""
        try:
            if not self.client:
                await self.connect()
            
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """키 존재 여부 확인"""
        try:
            if not self.client:
                await self.connect()
            
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"캐시 키 확인 실패: {e}")
            return False        
    