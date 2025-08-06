import pandas as pd
import numpy as np
import pickle
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# 텍스트 처리
import re
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# 고급 임베딩 (선택적)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers가 설치되지 않았습니다. 기본 임베딩을 사용합니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedContentPreprocessor:
    """고도화된 콘텐츠 전처리 클래스"""
    
    def __init__(self, use_korean_tokenizer: bool = True):
        self.use_korean_tokenizer = use_korean_tokenizer
        self.okt = Okt() if use_korean_tokenizer else None
        
        # 불용어 정의
        self.stop_words = {
            '이', '그', '저', '것', '수', '등', '및', '또는', '그리고', '하지만',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'
        }
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text)
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_korean(self, text: str) -> List[str]:
        """한국어 토큰화"""
        if not self.use_korean_tokenizer or not text:
            return text.split()
        
        try:
            # 명사만 추출 (가장 안정적)
            nouns = self.okt.nouns(text)
            
            # 품사 분석으로 명사, 형용사, 동사 추출
            pos_result = self.okt.pos(text, norm=True, stem=True)
            pos_tokens = []
            for token, pos in pos_result:
                if pos in ['Noun', 'Adjective', 'Verb']:
                    pos_tokens.append(token)
            
            # 모든 토큰 결합
            all_tokens = nouns + pos_tokens
            
            # 불용어 제거 및 길이 필터링
            tokens = [token for token in all_tokens 
                     if token not in self.stop_words and len(token) > 1]
            
            return tokens
        except Exception as e:
            logger.warning(f"한국어 토큰화 실패: {e}")
            return text.split()
    
    def preprocess_text(self, text: str) -> str:
        """전체 텍스트 전처리 파이프라인"""
        text = self.clean_text(text)
        if self.use_korean_tokenizer:
            tokens = self.tokenize_korean(text)
            return ' '.join(tokens)
        return text


class HybridRecommender:
    """하이브리드 추천 시스템"""
    
    def __init__(self, 
                 content_weight: float = 0.6,
                 collaborative_weight: float = 0.4,
                 use_bert: bool = False):
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.use_bert = use_bert and SENTENCE_TRANSFORMERS_AVAILABLE
        
        # 모델 컴포넌트
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.svd_model = None
        self.nmf_model = None
        self.kmeans_model = None
        self.bert_model = None
        
        # 유사도 행렬
        self.content_similarity = None
        self.collaborative_similarity = None
        self.hybrid_similarity = None
        
        # 메타데이터
        self.model_metadata = {}
        
    def fit_tfidf(self, corpus: List[str]) -> np.ndarray:
        """TF-IDF 벡터화"""
        logger.info("TF-IDF 벡터화 시작...")
        
        # 데이터 크기에 따라 파라미터 조정
        n_docs = len(corpus)
        min_df = max(1, min(2, n_docs // 2))  # 최소 1, 최대 2
        max_df = min(0.9, max(0.5, (n_docs - 1) / n_docs))  # 적절한 범위로 조정
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=min(2000, n_docs * 10),  # 데이터 크기에 맞게 조정
            ngram_range=(1, 2),  # 데이터가 적을 때는 bigram까지만
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            sublinear_tf=True
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        logger.info(f"TF-IDF 벡터 크기: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def fit_bert_embeddings(self, corpus: List[str]) -> np.ndarray:
        """BERT 임베딩 (선택적)"""
        if not self.use_bert:
            return None
        
        logger.info("BERT 임베딩 시작...")
        
        try:
            # 한국어에 특화된 BERT 모델 사용
            self.bert_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
            embeddings = self.bert_model.encode(corpus, show_progress_bar=True)
            logger.info(f"BERT 임베딩 크기: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"BERT 임베딩 실패: {e}")
            return None
    
    def fit_dimensionality_reduction(self, matrix: np.ndarray, method: str = 'svd') -> np.ndarray:
        """차원 축소"""
        logger.info(f"{method.upper()} 차원 축소 시작...")
        
        n_components = min(100, matrix.shape[1] // 2)
        
        if method == 'svd':
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = self.svd_model.fit_transform(matrix)
        elif method == 'nmf':
            self.nmf_model = NMF(n_components=n_components, random_state=42)
            reduced_matrix = self.nmf_model.fit_transform(matrix)
        else:
            return matrix
        
        logger.info(f"차원 축소 후 크기: {reduced_matrix.shape}")
        return reduced_matrix
    
    def fit_clustering(self, matrix: np.ndarray, n_clusters: int = 10) -> np.ndarray:
        """클러스터링"""
        # 데이터가 너무 적으면 클러스터링 건너뛰기
        if len(matrix) < 3:
            logger.info("데이터가 너무 적어 클러스터링을 건너뜁니다.")
            return np.zeros(len(matrix))
        
        # 클러스터 수 조정 (데이터 수보다 작게)
        n_clusters = min(n_clusters, len(matrix) - 1)
        if n_clusters < 2:
            logger.info("클러스터 수가 부족하여 클러스터링을 건너뜁니다.")
            return np.zeros(len(matrix))
        
        logger.info(f"K-means 클러스터링 시작 (클러스터 수: {n_clusters})...")
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(matrix)
        
        # 클러스터 품질 평가 (클러스터가 2개 이상일 때만)
        if len(np.unique(cluster_labels)) >= 2:
            try:
                silhouette_avg = silhouette_score(matrix, cluster_labels)
                logger.info(f"클러스터 품질 (Silhouette Score): {silhouette_avg:.3f}")
            except Exception as e:
                logger.warning(f"클러스터 품질 평가 실패: {e}")
        else:
            logger.info("클러스터가 1개뿐이어서 품질 평가를 건너뜁니다.")
        
        return cluster_labels
    
    def calculate_similarity(self, matrix: np.ndarray, method: str = 'cosine') -> np.ndarray:
        """유사도 계산"""
        logger.info(f"{method} 유사도 계산 시작...")
        
        if method == 'cosine':
            similarity = cosine_similarity(matrix)
        elif method == 'euclidean':
            # 유클리드 거리를 유사도로 변환
            distance = euclidean_distances(matrix)
            similarity = 1 / (1 + distance)
        else:
            similarity = cosine_similarity(matrix)
        
        logger.info(f"유사도 행렬 크기: {similarity.shape}")
        return similarity
    
    def create_hybrid_similarity(self, 
                                content_sim: np.ndarray, 
                                collaborative_sim: Optional[np.ndarray] = None) -> np.ndarray:
        """하이브리드 유사도 생성"""
        logger.info("하이브리드 유사도 생성...")
        
        if collaborative_sim is None:
            self.hybrid_similarity = content_sim
        else:
            # 가중 평균으로 하이브리드 유사도 계산
            self.hybrid_similarity = (
                self.content_weight * content_sim + 
                self.collaborative_weight * collaborative_sim
            )
        
        return self.hybrid_similarity


class AdvancedContentTrainer:
    """고도화된 콘텐츠 학습 클래스"""
    
    def __init__(self, 
                 data_path: str = "app/data",
                 model_path: str = "app/models",
                 use_korean: bool = True,
                 use_bert: bool = False):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.use_korean = use_korean
        self.use_bert = use_bert
        
        # 컴포넌트 초기화
        self.preprocessor = AdvancedContentPreprocessor(use_korean_tokenizer=use_korean)
        self.recommender = HybridRecommender(use_bert=use_bert)
        
        # 데이터
        self.content_df = None
        self.options_df = None
        self.processed_df = None
        
        # 모델 메타데이터
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '2.0.0',
            'features': [],
            'model_components': [],
            'performance_metrics': {}
        }
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        logger.info("데이터 로드 시작...")
        
        # 콘텐츠 데이터 로드
        content_file = self.data_path / "content.csv"
        options_file = self.data_path / "content_options.csv"
        
        if not content_file.exists():
            raise FileNotFoundError(f"콘텐츠 파일을 찾을 수 없습니다: {content_file}")
        
        self.content_df = pd.read_csv(content_file)
        logger.info(f"콘텐츠 데이터 로드 완료: {len(self.content_df)}개")
        
        # 옵션 데이터 로드 (있는 경우)
        if options_file.exists():
            self.options_df = pd.read_csv(options_file)
            logger.info(f"옵션 데이터 로드 완료: {len(self.options_df)}개")
            
            # 옵션 데이터 집계
            options_agg = self.options_df.groupby('content_id').agg({
                'name': lambda x: ' '.join(x.astype(str)),
                'description': lambda x: ' '.join(x.dropna().astype(str))
            }).reset_index()
            
            # 데이터 병합
            self.content_df = self.content_df.merge(
                options_agg, how='left', left_on='id', right_on='content_id', 
                suffixes=('', '_opt')
            )
        
        return self.content_df
    
    def preprocess_data(self) -> pd.DataFrame:
        """데이터 전처리"""
        logger.info("데이터 전처리 시작...")
        
        df = self.content_df.copy()
        
        # 텍스트 필드 전처리
        text_fields = ['title', 'maker_intro', 'service_process', 'service_target', 
                      'content_introduction', 'name', 'description']
        
        for field in text_fields:
            if field in df.columns:
                df[f'{field}_processed'] = df[field].apply(self.preprocessor.preprocess_text)
        
        # 수치형 특성 생성
        df['text_length'] = df['title'].str.len()
        df['word_count'] = df['title'].str.split().str.len()
        
        # 카테고리형 특성 인코딩
        if 'content_type' in df.columns:
            df['content_type_encoded'] = pd.Categorical(df['content_type']).codes
        
        self.processed_df = df
        logger.info("데이터 전처리 완료")
        
        return df
    
    def create_corpus(self) -> List[str]:
        """텍스트 코퍼스 생성"""
        logger.info("텍스트 코퍼스 생성...")
        
        corpus = []
        for _, row in self.processed_df.iterrows():
            text_parts = []
            
            # 전처리된 텍스트 필드들
            processed_fields = [col for col in self.processed_df.columns if col.endswith('_processed')]
            
            for field in processed_fields:
                if pd.notna(row[field]) and row[field].strip():
                    text_parts.append(row[field])
            
            # 원본 필드들도 추가 (전처리된 것이 없는 경우)
            original_fields = ['title', 'maker_intro', 'content_introduction']
            for field in original_fields:
                if field in row and pd.notna(row[field]) and not any(f'{field}_processed' in col for col in processed_fields):
                    text_parts.append(self.preprocessor.preprocess_text(row[field]))
            
            combined_text = ' '.join(text_parts)
            corpus.append(combined_text)
        
        logger.info(f"코퍼스 생성 완료: {len(corpus)}개 문서")
        return corpus
    
    def train_content_based_model(self, corpus: List[str]) -> np.ndarray:
        """콘텐츠 기반 모델 학습"""
        logger.info("콘텐츠 기반 모델 학습 시작...")
        
        # TF-IDF 벡터화
        tfidf_matrix = self.recommender.fit_tfidf(corpus)
        
        # BERT 임베딩 (선택적)
        bert_matrix = None
        if self.use_bert:
            bert_matrix = self.recommender.fit_bert_embeddings(corpus)
        
        # 차원 축소
        reduced_tfidf = self.recommender.fit_dimensionality_reduction(tfidf_matrix, 'svd')
        
        # 클러스터링
        cluster_labels = self.recommender.fit_clustering(reduced_tfidf, n_clusters=min(10, len(corpus)//2))
        
        # 유사도 계산
        content_similarity = self.recommender.calculate_similarity(reduced_tfidf, 'cosine')
        
        # 메타데이터 업데이트
        self.metadata['features'].extend(['tfidf', 'svd', 'clustering'])
        self.metadata['model_components'].append('content_based')
        
        return content_similarity
    
    def create_wordcloud(self, corpus: List[str], save_path: str = None):
        """워드클라우드 생성"""
        logger.info("워드클라우드 생성...")
        
        try:
            # matplotlib 백엔드를 Agg로 설정 (GUI 없이)
            import matplotlib
            matplotlib.use('Agg')
            
            text = ' '.join(corpus)
            wordcloud = WordCloud(
                font_path='/System/Library/Fonts/AppleGothic.ttf',  # macOS 한글 폰트
                width=800, 
                height=400,
                background_color='white',
                max_words=100
            ).generate(text)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('콘텐츠 키워드 워드클라우드')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"워드클라우드 저장: {save_path}")
            
            plt.close()  # GUI 창을 열지 않고 바로 닫기
        except Exception as e:
            logger.warning(f"워드클라우드 생성 실패: {e}")
    
    def evaluate_model(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """모델 평가"""
        logger.info("모델 평가 시작...")
        
        metrics = {}
        
        # 유사도 분포 분석
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        metrics['mean_similarity'] = np.mean(similarities)
        metrics['std_similarity'] = np.std(similarities)
        metrics['max_similarity'] = np.max(similarities)
        metrics['min_similarity'] = np.min(similarities)
        
        # 다양성 측정 (유사도 표준편차)
        metrics['diversity_score'] = 1 - metrics['std_similarity']
        
        # 밀도 측정 (높은 유사도 비율)
        high_similarity_ratio = np.sum(similarities > 0.5) / len(similarities)
        metrics['density_score'] = high_similarity_ratio
        
        logger.info(f"모델 평가 완료: {metrics}")
        return metrics
    
    def save_model(self, similarity_matrix: np.ndarray, corpus: List[str]):
        """모델 저장"""
        logger.info("모델 저장 시작...")
        
        # 모델 디렉토리 생성
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # 모델 데이터 준비
        model_data = {
            'similarity_matrix': similarity_matrix,
            'content_data': self.processed_df[['id', 'title', 'maker_intro', 'content_introduction']],
            'vectorizer': self.recommender.tfidf_vectorizer,
            'svd_model': self.recommender.svd_model,
            'kmeans_model': self.recommender.kmeans_model,
            'bert_model': self.recommender.bert_model,
            'corpus': corpus,
            'metadata': self.metadata
        }
        
        # 메인 모델 저장
        model_file = self.model_path / "content_sim.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 개별 컴포넌트 저장 (선택적)
        components_dir = self.model_path / "components"
        components_dir.mkdir(exist_ok=True)
        
        joblib.dump(self.recommender.tfidf_vectorizer, components_dir / "tfidf_vectorizer.joblib")
        joblib.dump(self.recommender.svd_model, components_dir / "svd_model.joblib")
        joblib.dump(self.recommender.kmeans_model, components_dir / "kmeans_model.joblib")
        
        # 메타데이터 JSON 저장
        metadata_file = self.model_path / "model_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"모델 저장 완료: {model_file}")
    
    def train(self):
        """전체 학습 파이프라인"""
        logger.info("=== 고도화된 콘텐츠 추천 모델 학습 시작 ===")
        
        try:
            # 1. 데이터 로드
            self.load_data()
            
            # 2. 데이터 전처리
            self.preprocess_data()
            
            # 3. 코퍼스 생성
            corpus = self.create_corpus()
            
            # 4. 워드클라우드 생성 (시각화)
            wordcloud_path = self.model_path / "wordcloud.png"
            self.create_wordcloud(corpus, str(wordcloud_path))
            
            # 5. 콘텐츠 기반 모델 학습
            similarity_matrix = self.train_content_based_model(corpus)
            
            # 6. 모델 평가
            performance_metrics = self.evaluate_model(similarity_matrix)
            self.metadata['performance_metrics'] = performance_metrics
            
            # 7. 모델 저장
            self.save_model(similarity_matrix, corpus)
            
            logger.info("=== 학습 완료 ===")
            logger.info(f"총 {len(self.content_df)}개의 콘텐츠에 대한 고도화된 추천 모델을 생성했습니다.")
            logger.info(f"성능 지표: {performance_metrics}")
            
        except Exception as e:
            logger.error(f"학습 중 오류 발생: {e}")
            raise


def main():
    """메인 실행 함수"""
    # 학습 설정
    trainer = AdvancedContentTrainer(
        data_path="app/data",
        model_path="app/models",
        use_korean=True,
        use_bert=False  # BERT 사용 시 sentence-transformers 설치 필요
    )
    
    # 학습 실행
    trainer.train()


if __name__ == "__main__":
    main()