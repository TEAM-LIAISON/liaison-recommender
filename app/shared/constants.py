# app/shared/constants.py
"""상수 정의"""

# 캐시 키 패턴
CACHE_KEYS = {
    "RECOMMENDATIONS": "recommendations:{user_id}",
    "USER_PREFERENCES": "user_preferences:{user_id}",
    "CONTENT_FEATURES": "content_features:{content_id}",
    "MODEL_METADATA": "model_metadata"
}

# 추천 알고리즘 가중치
ALGORITHM_WEIGHTS = {
    "content_based": 0.4,
    "collaborative": 0.3,
    "clustering": 0.2,
    "hybrid": 0.1
}

# 기본 설정값
DEFAULT_RECOMMENDATION_LIMIT = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_CACHE_TTL = 3600