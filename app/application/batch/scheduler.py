import logging
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from ...config import settings

class RecommendationScheduler:
    """추천 모델 자동 재학습 스케줄러"""

    def __init__(self, model_loader):
        self.logger = logging.getLogger(__name__)
        self.model_loader = model_loader

        # AsyncIO 스케줄러 인스턴스 생성
        self.scheduler = AsyncIOScheduler(timezone=settings.server.timezone)

        # 매일 새벽 03:00에 모델 재로드 작업 등록
        self.scheduler.add_job(
            self._reload_models,
            CronTrigger(hour=3, minute=0),
            id="recommendation_reload",
            name="Daily reload recommendation models",
            replace_existing=True,
        )

        # 스케줄러 시작
        self.scheduler.start()
        self.logger.info("RecommendationScheduler started: models reload scheduled at 03:00 %s", settings.server.timezone)

    async def _reload_models(self):
        """스케줄에 의해 호출되어 추천 모델을 재로드합니다"""
        self.logger.info("Scheduled task: reloading recommendation models...")
        try:
            # 모델 로더의 비동기 재로드 메서드 호출
            await self.model_loader.load_all_models()
            self.logger.info("Recommendation models reloaded successfully.")
        except Exception as e:
            self.logger.error("Failed to reload recommendation models: %s", e, exc_info=True)
