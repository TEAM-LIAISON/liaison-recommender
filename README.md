
## 프로젝트 구조

```bash

.
├── app/
│   ├── __init__.py
│   ├── main.py                           # FastAPI 앱 진입점
│   ├── config.py                         # 환경 설정
│   ├── dependencies.py                   # 의존성 주입
│   │
│   ├── core/                             # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── recommenders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # 추천 인터페이스
│   │   │   ├── content_based.py         # TF-IDF 기반
│   │   │   ├── collaborative.py         # SVD 기반
│   │   │   ├── clustering.py            # K-means 기반
│   │   │   └── hybrid.py                # 앙상블
│   │   └── scoring/
│   │       ├── __init__.py
│   │       └── ranker.py                # 점수 계산/랭킹
│   │
│   ├── application/                      # 애플리케이션 서비스
│   │   ├── __init__.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── recommendation_service.py
│   │   │   └── cache_service.py
│   │   └── batch/
│   │       ├── __init__.py
│   │       ├── batch_recommender.py     # 배치 추천 생성
│   │       └── scheduler.py             # 스케줄러
│   │
│   ├── infrastructure/                   # 외부 시스템 연동
│   │   ├── __init__.py
│   │   ├── clients/
│   │   │   ├── __init__.py
│   │   │   └── spring_boot_client.py   # Spring Boot API 클라이언트
│   │   ├── cache/
│   │   │   ├── __init__.py
│   │   │   └── redis_client.py         # Redis 캐시
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── model_loader.py         # 모델 로딩
│   │   │   └── data_source.py          # 데이터 소스 관리
│   │   └── models/                      # 학습된 모델 파일
│   │       ├── tfidf_vectorizer.joblib
│   │       ├── svd_model.joblib
│   │       ├── kmeans_model.joblib
│   │       ├── content_sim.pkl
│   │       └── model_metadata.json
│   │
│   ├── api/                             # REST API
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── recommendations.py  # 추천 엔드포인트
│   │   │   │   └── health.py          # 헬스체크
│   │   │   └── schemas/
│   │   │       ├── __init__.py
│   │   │       ├── request.py         # 요청 모델
│   │   │       └── response.py        # 응답 모델
│   │   └── middleware/
│   │       ├── __init__.py
│   │       └── error_handler.py       # 에러 처리
│   │
│   └── shared/                          # 공통 모듈
│       ├── __init__.py
│       ├── enums.py                    # Enum 정의
│       ├── exceptions.py               # 커스텀 예외
│       └── constants.py                # 상수
│
├── scripts/                             # 실행 스크립트
│   ├── train.py                        # 모델 학습
│   └── evaluate.py                     # 모델 평가
│
├── tests/                              # 테스트
│   ├── __init__.py
│   ├── unit/
│   └── integration/
│
├── .env                                # 환경변수 예시
├── requirements.txt                    # 의존성
├── docker-compose.yml                  # Docker 설정
└── README.md                          # 문서
```