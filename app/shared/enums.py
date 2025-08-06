from enum import Enum


class ContentType(str, Enum):
    """콘텐츠 유형 정의"""
    COACHING = "coaching"
    DOCUMENT = "document"


class ContentStatus(str, Enum):
    """콘텐츠 상태 정의"""
    DRAFT = "draft"
    ACTIVE = "active"
    DELETED = "deleted"
    DISCONTINUED = "discontinued" 

class CacheType(str, Enum):
    """캐시 타입 정의"""
    MEMORY = "memory"
    REDIS = "redis"

class RecommendationType(str, Enum):
    """추천 알고리즘 타입 정의"""
    CONTENT_BASED = "content_based"
    COLLABORATIVE = "collaborative"
    CLUSTERING = "clustering"
    HYBRID = "hybrid"