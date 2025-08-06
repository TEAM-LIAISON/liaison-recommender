# app/infrastructure/clients/spring_boot_client.py
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from ...shared.exceptions import APIClientException

logger = logging.getLogger(__name__)

class SpringBootClient:
    """Spring Boot API 클라이언트"""

    def __init__(self, base_url: str, timeout: int = 30, retry_attempts: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.session = None

        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """API 요청 실행"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        logger.warning(f"리소스를 찾을 수 없습니다: {url}")
                        return None
                    else:
                        logger.error(f"API 요청 실패: {response.status} - {url}")
                        if attempt == self.retry_attempts - 1:
                            raise APIClientException(f"API 요청 실패: {response.status}")
                        
            except Exception as e:
                logger.warning(f"API 요청 실패 (시도 {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt == self.retry_attempts - 1:
                    raise APIClientException(f"API 요청 실패: {e}")
        
        return None
    
    # 콘텐츠 관련 API
    async def get_all_contents(self) -> List[Dict]:
        """모든 활성 콘텐츠 조회"""
        response = await self._make_request('GET', '/api/v1/contents/active')
        return response.get('data', []) if response else []
    
    async def get_content_by_id(self, content_id: int) -> Optional[Dict]:
        """ID로 콘텐츠 조회"""
        response = await self._make_request('GET', f'/api/v1/contents/{content_id}')
        return response.get('data') if response else None
    
    async def get_content_features(self, content_id: int) -> Optional[Dict]:
        """콘텐츠 특성 조회"""
        response = await self._make_request('GET', f'/api/v1/contents/{content_id}/features')
        return response.get('data') if response else None
    
    # 사용자 관련 API
    async def get_user_purchases(self, user_id: int) -> List[Dict]:
        """사용자 구매 이력"""
        response = await self._make_request('GET', f'/api/v1/users/{user_id}/purchases')
        return response.get('data', []) if response else []
    
    async def get_user_views(self, user_id: int) -> List[Dict]:
        """사용자 조회 이력"""
        response = await self._make_request('GET', f'/api/v1/users/{user_id}/views')
        return response.get('data', []) if response else []
    
    async def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """사용자 선호도"""
        response = await self._make_request('GET', f'/api/v1/users/{user_id}/preferences')
        return response.get('data') if response else None
    
    # 구매 관련 API
    async def get_all_purchases(self) -> List[Dict]:
        """모든 구매 이력"""
        response = await self._make_request('GET', '/api/v1/purchases')
        return response.get('data', []) if response else []
