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
