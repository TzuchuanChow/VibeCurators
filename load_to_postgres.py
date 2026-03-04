"""
VibeLens PostgreSQL Loader
Load data with embeddings into Aiven PostgreSQL + pgvector
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import time
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PostgreSQLLoader:
    def __init__(self, db_config):
        """
        Initialize PostgreSQL Loader
        
        Args:
            db_config: Database connection configuration dictionary
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to PostgreSQL"""
        print("=== Connecting to PostgreSQL ===")
        print(f"Host: {self.db_config['host']}")
        print(f"Database: {self.db_config['database']}")
        
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                sslmode=self.db_config.get('sslmode', 'require')
            )
            self.cursor = self.conn.cursor()
            
            # Enable pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
            
            print("Connection successful!")
            
        except Exception as e:
            print(f"Connection failed: {e}")
            raise
    
    def create_table(self, embedding_dim=None):
        """创建 movies 表"""
        # 如果传入了参数，使用传入的值；否则使用实例变量
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        
        print("\n=== Creating Table ===")
        print(f"Embedding dimension: {self.embedding_dim}")
        # 删除旧表
        drop_table_sql = "DROP TABLE IF EXISTS movies CASCADE;"
        self.cursor.execute(drop_table_sql)
        
        # 创建表
        create_table_sql = f"""
        CREATE TABLE movies (
            "movieId" INTEGER PRIMARY KEY,
            "title" TEXT NOT NULL,
            "year" INTEGER,
            "genres" TEXT,
            "tmdb_genres" TEXT,
            "num_ratings" INTEGER,
            "avg_rating" REAL,
            "tmdb_rating" REAL,
            "soup" TEXT,
            "embedding" VECTOR({self.embedding_dim})
        );
        """
        
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        print("Table 'movies' created successfully!")
    
    def load_data(self, parquet_path):
        """
        Load data from Parquet
        
        Returns:
            df: Pandas DataFrame
        """
        print(f"\n=== Loading Data from Parquet ===")
        print(f"Path: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        
        print(f"Loaded {len(df):,} movies")
        
        # Verify required fields
        required_columns = ['movieId', 'title', 'soup', 'embedding']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        return df
    
    def bulk_insert_copy(self, df):
        """
        Bulk insert using COPY (fastest method)
        
        Args:
            df: Pandas DataFrame with embeddings
        """
        print(f"\n=== Bulk Insert (COPY method) ===")
        print(f"Inserting {len(df):,} rows...")
        
        start_time = time.time()
        
        # Prepare data
        # Convert embedding list to PostgreSQL vector format string
        df_copy = df.copy()
        df_copy['embedding_str'] = df_copy['embedding'].apply(
            lambda x: '[' + ','.join(map(str, x)) + ']'
        )
        
        # Select columns to insert
        columns_to_insert = [
            'movieId', 'title', 'year', 'genres', 'tmdb_genres',
            'num_ratings', 'avg_rating', 'tmdb_rating', 'soup', 'embedding_str'
        ]
        
        # Handle missing values
        df_copy = df_copy[columns_to_insert].fillna({
            'year': 0,
            'genres': '',
            'tmdb_genres': '',
            'num_ratings': 0,
            'avg_rating': 0.0,
            'tmdb_rating': 0.0,
            'soup': ''
        })
        
        # Create StringIO buffer
        buffer = StringIO()
        df_copy.to_csv(buffer, index=False, header=False, sep='\t')
        buffer.seek(0)
        
        # Execute COPY
        try:
            self.cursor.copy_from(
                buffer,
                'movies',
                sep='\t',
                columns=['movie_id', 'title', 'year', 'genres', 'tmdb_genres',
                        'num_ratings', 'avg_rating', 'tmdb_rating', 'soup', 'embedding'],
                null=''
            )
            self.conn.commit()
            
            elapsed_time = time.time() - start_time
            print(f"Insert complete in {elapsed_time:.2f}s")
            print(f"Speed: {len(df)/elapsed_time:.0f} rows/second")
            
        except Exception as e:
            self.conn.rollback()
            print(f"COPY failed: {e}")
            print("Falling back to execute_values method...")
            self.bulk_insert_execute_values(df)
        
    def bulk_insert_execute_values(self, df):
            """使用 execute_values 批量插入（修复版本）"""
            print("\n=== Bulk Insert (execute_values method) ===")
            
            batch_size = 1000
            print(f"Batch size: {batch_size}")
            
            # SQL 插入语句
            insert_sql = """
                        INSERT INTO movies (
                            "movieId", "title", "year", "genres", "tmdb_genres",
                            "num_ratings", "avg_rating", "tmdb_rating", "soup", "embedding"
                        ) VALUES %s
                        ON CONFLICT ("movieId") DO NOTHING
                    """
            
            # 准备数据 - 关键修复
            data = []
            for idx, row in df.iterrows():
                try:
                    # 1. 转换 embedding: numpy.ndarray → Python list
                    embedding = row['embedding'].tolist()
                    
                    # 2. 清理文本字段（移除 NULL 字符和处理 None）
                    title = str(row['title']).replace('\x00', '') if pd.notna(row['title']) else ''
                    soup = str(row['soup']).replace('\x00', '') if pd.notna(row['soup']) else ''
                    genres = str(row['genres']).replace('\x00', '') if pd.notna(row['genres']) else ''
                    tmdb_genres = str(row['tmdb_genres']).replace('\x00', '') if pd.notna(row['tmdb_genres']) else ''
                    
                    # 3. 准备行数据
                    data.append((
                        int(row['movieId']),
                        title,
                        int(row['year']) if pd.notna(row['year']) else None,
                        genres,
                        tmdb_genres,
                        int(row['num_ratings']),
                        float(row['avg_rating']),
                        float(row['tmdb_rating']) if pd.notna(row['tmdb_rating']) else None,
                        soup,
                        embedding  # 现在是 Python list
                    ))
                except Exception as e:
                    print(f"Warning: Skipping row {idx} (movieId={row['movieId']}): {e}")
                    continue
            
            print(f"Prepared {len(data)} rows for insertion")
            
            # 批量插入
            from psycopg2.extras import execute_values
            import time
            
            start = time.time()
            try:
                execute_values(self.cursor, insert_sql, data, page_size=batch_size)
                self.conn.commit()
                elapsed = time.time() - start
                
                print(f"Insert complete in {elapsed:.2f}s")
                print(f"Speed: {len(data)/elapsed:.0f} rows/second")
            except Exception as e:
                print(f"Error during bulk insert: {e}")
                self.conn.rollback()
                raise
                
    def create_hnsw_index(self):
            """创建 HNSW 索引"""
            print("\n=== Creating HNSW Index ===")
            
            m = 32
            ef_construction = 128
            
            print(f"Parameters: m={m}, ef_construction={ef_construction}")
            print("This may take a few minutes...")
            
            import time
            start = time.time()
            
            # 创建索引 SQL
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS movies_embedding_idx 
            ON movies 
            USING hnsw ("embedding" vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
            """
            
            # 执行
            self.cursor.execute(create_index_sql)  # ← 确保变量名正确
            self.conn.commit()
            
            elapsed = time.time() - start
            print(f"Index created in {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
            
            # 验证索引
            self.cursor.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'movies'
            """)
            indexes = self.cursor.fetchall()
            print(f"Indexes on 'movies' table: {len(indexes)}")
            for idx in indexes:
                print(f"  - {idx[0]}")

    def create_query_cache_table(self):
        """创建查询缓存表"""
        print("\n=== Creating Query Cache Table ===")
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS query_cache (
            query_text TEXT PRIMARY KEY,
            embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            hit_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS query_cache_hnsw_idx 
        ON query_cache 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        
        CREATE INDEX IF NOT EXISTS query_cache_hit_count_idx 
        ON query_cache (hit_count DESC);
        """
        
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        print("✅ Query cache table created")

    def verify_data(self):
            """验证数据加载成功"""
            print("\n=== Verifying Data ===")
            
            # 检查总数
            self.cursor.execute('SELECT COUNT(*) FROM movies')
            count = self.cursor.fetchone()[0]
            print(f"Total movies in database: {count:,}")
            
            # 显示样本记录
            self.cursor.execute("""
                SELECT "movieId", "title", "year", "num_ratings"
                FROM movies
                ORDER BY "num_ratings" DESC
                LIMIT 5
            """)
            
            print("\nSample records:")
            for row in self.cursor.fetchall():
                movie_id, title, year, num_ratings = row
                print(f"  ID {movie_id}: {title} ({year}) - {num_ratings:,} ratings")
            
            # 验证 embedding 维度
            self.cursor.execute("""
                SELECT vector_dims("embedding") 
                FROM movies 
                LIMIT 1
            """)
            dim = self.cursor.fetchone()[0]
            print(f"\nEmbedding dimension check: {dim}")
    
    def test_search(self, query_text, top_k=5):
        """
        Test vector search
        
        Args:
            query_text: Query text
            top_k: Return top K results
        """
        print(f"\n=== Testing Vector Search ===")
        print(f"Query: '{query_text}'")
        
        # Note: This requires Sentence-Transformers to encode query
        print("Note: This requires Sentence-Transformers to encode the query")
        print("Use the search_movies.py script for actual searches")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("\nDatabase connection closed")
    
    def run_pipeline(self, parquet_path, embedding_dim=384):
        """
        Run complete data import pipeline
        
        Args:
            parquet_path: Parquet file with embeddings
            embedding_dim: Embedding dimension
        """
        total_start = time.time()
        
        try:
            # 1. Connect to database
            self.connect()
            
            # 2. Create table
            self.create_table(embedding_dim=embedding_dim)
            
            # 3. Load data
            df = self.load_data(parquet_path)
            
            # 4. Bulk insert
            self.bulk_insert_copy(df)
            
            # 5. Create index
            self.create_hnsw_index()

            self.create_query_cache_table() 
            
            # 6. Verify
            self.verify_data()
            
            total_time = time.time() - total_start
            print(f"\n{'='*50}")
            print(f"Database Loading Complete!")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"{'='*50}")
            
        finally:
            self.close()


if __name__ == "__main__":
    # Database configuration (read from environment variables)
    db_config = {
        'host': os.getenv('PG_HOST', 'your-aiven-host.aivencloud.com'),
        'port': int(os.getenv('PG_PORT', 12345)),
        'database': os.getenv('PG_DATABASE', 'defaultdb'),
        'user': os.getenv('PG_USER', 'avnadmin'),
        'password': os.getenv('PG_PASSWORD', 'your-password'),
        'sslmode': 'require'
    }
    
    # Input file
    INPUT_PARQUET = './data/preprocessed/movies_with_embeddings.parquet'
    
    # Embedding dimension (must match generation)
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
    
    # Run
    loader = PostgreSQLLoader(db_config)
    loader.run_pipeline(
        parquet_path=INPUT_PARQUET,
        embedding_dim=EMBEDDING_DIM
    )
