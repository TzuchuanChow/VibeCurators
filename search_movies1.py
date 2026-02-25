"""
VibeLens Movie Search
Semantic search based on user descriptions
"""

import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class MovieSearchEngine:
    def __init__(self, db_config, model_name='all-MiniLM-L6-v2'):
        """
        Initialize search engine
        
        Args:
            db_config: Database configuration
            model_name: Sentence-Transformers model name
        """
        self.db_config = db_config
        self.model_name = model_name
        self.model = None
        self.conn = None
        
    def load_model(self):
        """Load Sentence-Transformers model"""
        print(f"Loading model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded!")
    
    def connect_db(self):
        """Connect to database"""
        self.conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            sslmode=self.db_config.get('sslmode', 'require')
        )
    
    def search_movies(self, query_text, top_k=5, year_filter=None):
        """
        Semantic movie search
        
        Args:
            query_text: User input description
            top_k: Return top K results
            year_filter: Year filter (e.g., ">= 2000")
            
        Returns:
            results: List with (title, year, genres, similarity) per item
        """
        # 1. Convert query to embedding
        query_embedding = self.model.encode([query_text])[0]
        
        # 2. Build SQL
        sql = """
        SELECT 
            title,
            year,
            tmdb_genres,
            num_ratings,
            avg_rating,
            1 - (embedding <=> %s::vector) AS similarity
        FROM movies
        """
        
        # Add year filter
        where_clauses = []
        params = [query_embedding.tolist()]
        
        if year_filter:
            where_clauses.append(f"year {year_filter}")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        sql += """
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        params.extend([query_embedding.tolist(), top_k])
        
        # 3. Execute query
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def print_results(self, results, query_text):
        """Format and print search results"""
        print(f"\n{'='*70}")
        print(f"Search Query: '{query_text}'")
        print(f"{'='*70}\n")
        
        for i, (title, year, genres, num_ratings, avg_rating, similarity) in enumerate(results, 1):
            print(f"{i}. {title} ({year})")
            print(f"   Genres: {genres}")
            print(f"   Ratings: {num_ratings:,} reviews, avg {avg_rating:.1f}/5.0")
            print(f"   Similarity: {similarity:.3f}")
            print()
    
    def interactive_search(self):
        """Interactive search mode"""
        print("\n=== VibeLens Interactive Search ===")
        print("Enter your movie vibe description (or 'quit' to exit)\n")
        
        while True:
            query = input("Search > ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            # Ask for year filter
            year_filter_input = input("Year filter (e.g., '>= 2000', or press Enter to skip): ").strip()
            year_filter = year_filter_input if year_filter_input else None
            
            # Execute search
            results = self.search_movies(query, top_k=5, year_filter=year_filter)
            
            # Print results
            self.print_results(results, query)
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': os.getenv('PG_HOST', 'your-aiven-host.aivencloud.com'),
        'port': int(os.getenv('PG_PORT', 12345)),
        'database': os.getenv('PG_DATABASE', 'defaultdb'),
        'user': os.getenv('PG_USER', 'avnadmin'),
        'password': os.getenv('PG_PASSWORD', 'your-password'),
        'sslmode': 'require'
    }
    
    # Initialize search engine
    search_engine = MovieSearchEngine(db_config)
    
    try:
        # Load model
        search_engine.load_model()
        
        # Connect to database
        search_engine.connect_db()
        
        # Start interactive search
        search_engine.interactive_search()
        
    finally:
        search_engine.close()
        print("\nGoodbye!")
