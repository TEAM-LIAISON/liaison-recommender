# app/data_source.py
import logging
from typing import Dict, List, Optional
import mysql.connector
from mysql.connector import pooling

logger = logging.getLogger(__name__)

class GrobleDataSource:
    """그로블 데이터베이스 연결 및 데이터 조회"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection_pool = self.create_connection_pool()
    
    def create_connection_pool(self):
        """데이터베이스 연결 풀 생성"""
        return mysql.connector.pooling.MySQLConnectionPool(
            pool_name="groble_pool",
            pool_size=5,
            **self.db_config
        )

    def get_connection(self):
        """연결 풀에서 연결 가져오기"""
        return self.connection_pool.get_connection()