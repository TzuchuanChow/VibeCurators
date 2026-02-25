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
    
    def create_table(self, embedding_dim=384):
        """
        Create movies table
        
        Args:
            embedding_dim: Embedding vector dimension
        """
        print(f"\n=== Creating Table ===")
        print(f"Embedding dimension: {embedding_dim}")
        
        # Drop old table if exists
        drop_table_sql = "DROP TABLE IF EXISTS movies CASCADE"
        self.cursor.execute(drop_table_sql)
        
        # Create new table
        create_table_sql = f"""
        CREATE TABLE movies (
            id SERIAL PRIMARY KEY,
            movie_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            year INTEGER,
            genres TEXT,
            tmdb_genres TEXT,
            num_ratings INTEGER,
            avg_rating FLOAT,
            tmdb_rating FLOAT,
            soup TEXT,
            embedding vector({embedding_dim}) NOT NULL
        )
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
    
    def bulk_insert_execute_values(self, df, batch_size=1000):
        """
        Bulk insert using execute_values (fallback method)
        
        Args:
            df: Pandas DataFrame
            batch_size: Number of rows per batch
        """
        print(f"\n=== Bulk Insert (execute_values method) ===")
        print(f"Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Prepare data
        data = []
        for _, row in df.iterrows():
            data.append((
                int(row['movieId']),
                str(row['title']),
                int(row.get('year', 0)) if pd.notna(row.get('year')) else None,
                str(row.get('genres', '')),
                str(row.get('tmdb_genres', '')),
                int(row.get('num_ratings', 0)),
                float(row.get('avg_rating', 0.0)),
                float(row.get('tmdb_rating', 0.0)),
                str(row.get('soup', '')),
                row['embedding']  # List format
            ))
        
        # Bulk insert
        insert_sql = """
        INSERT INTO movies (
            movie_id, title, year, genres, tmdb_genres,
            num_ratings, avg_rating, tmdb_rating, soup, embedding
        ) VALUES %s
        """
        
        execute_values(self.cursor, insert_sql, data, page_size=batch_size)
        self.conn.commit()
        
        elapsed_time = time.time() - start_time
        print(f"Insert complete in {elapsed_time:.2f}s")
        print(f"Speed: {len(df)/elapsed_time:.0f} rows/second")
    
    def create_hnsw_index(self, m=32, ef_construction=128):
        """
        Create HNSW index (vector approximate nearest neighbor search)
        
        Args:
            m: HNSW parameter - max connections per layer (recommended 16)
            ef_construction: Search range during index build (recommended 64)
        """
        print(f"\n=== Creating HNSW Index ===")
        print(f"Parameters: m={m}, ef_construction={ef_construction}")
        print("This may take a few minutes...")
        
        start_time = time.time()
        
        # Create HNSW index
        index_sql = f"""
        CREATE INDEX ON movies 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
        """
        
        self.cursor.execute(index_sql)
        self.conn.commit()
        
        elapsed_time = time.time() - start_time
        print(f"Index created in {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        
        # Verify index
        self.cursor.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'movies'
        """)
        indexes = self.cursor.fetchall()
        print(f"Indexes on 'movies' table: {len(indexes)}")
        for idx_name, idx_def in indexes:
            print(f"  - {idx_name}")
    
    def verify_data(self):
        """Verify imported data"""
        print(f"\n=== Verifying Data ===")
        
        # Count rows
        self.cursor.execute("SELECT COUNT(*) FROM movies")
        count = self.cursor.fetchone()[0]
        print(f"Total movies in database: {count:,}")
        
        # Show samples
        self.cursor.execute("""
            SELECT id, title, year, num_ratings 
            FROM movies 
            LIMIT 5
        """)
        samples = self.cursor.fetchall()
        print("\nSample records:")
        for record in samples:
            print(f"  ID {record[0]}: {record[1]} ({record[2]}) - {record[3]} ratings")
        
        # Check embedding dimension
        self.cursor.execute("""
            SELECT vector_dims(embedding) 
            FROM movies 
            LIMIT 1
        """)
        embedding_dim = self.cursor.fetchone()[0]
        print(f"\nEmbedding dimension: {embedding_dim}")
    
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
