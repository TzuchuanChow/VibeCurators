#!/usr/bin/env python3
"""
VibeLens - Semantic Movie Search Engine
Features: Persistent query caching, genre-specific HNSW index routing, cache analytics
"""

import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()


class MovieSearchEngine:
    """Semantic movie search with persistent query caching and genre optimization."""

    # Genre keywords for routing queries to genre-specific HNSW indexes
    GENRE_KEYWORDS = {
        'Science Fiction': ['sci-fi', 'science fiction', 'space', 'robot', 'ai',
                            'future', 'futuristic', 'alien', 'dystopia', 'cyberpunk',
                            'time travel', 'interstellar'],
        'Action': ['action', 'fight', 'explosion', 'car chase', 'combat',
                   'martial arts', 'battle', 'war', 'shootout'],
        'Romance': ['romantic', 'love', 'romance', 'relationship', 'love story',
                    'wedding', 'heartbreak'],
        'Comedy': ['funny', 'comedy', 'laugh', 'humor', 'humorous', 'hilarious',
                   'slapstick', 'satire', 'parody'],
        'Thriller': ['thriller', 'suspense', 'suspenseful', 'mystery', 'tense',
                     'psychological thriller', 'crime thriller'],
        'Horror': ['horror', 'scary', 'monster', 'ghost', 'haunted', 'zombie',
                   'slasher', 'supernatural', 'creepy'],
        'Drama': ['drama', 'dramatic', 'emotional', 'moving', 'powerful',
                  'heartfelt', 'tragic'],
        'Animation': ['animated', 'animation', 'cartoon', 'anime', 'pixar',
                      'studio ghibli'],
    }

    def __init__(self, db_config=None, model_name='all-MiniLM-L6-v2'):
        """
        Initialize search engine with database connection and embedding model.

        Args:
            db_config: Dict with keys: host, port, database, user, password.
                       If None, reads from environment variables.
            model_name: Sentence-transformer model (must match embedding generation).
        """
        if db_config is None:
            db_config = {
                'host': os.getenv('PG_HOST'),
                'port': os.getenv('PG_PORT', '5432'),
                'database': os.getenv('PG_DATABASE', 'vibelens'),
                'user': os.getenv('PG_USER', 'postgres'),
                'password': os.getenv('PG_PASSWORD'),
            }

        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

        # Session-level cache stats (complements persistent DB stats)
        self.cache_stats = {'hits': 0, 'misses': 0}

    # ── Query Embedding with Persistent Cache ──────────────────────────

    def get_query_embedding(self, query_text):
        """
        Get embedding from persistent DB cache, or compute and store it.

        Cache hit:  ~1ms (DB lookup only)
        Cache miss: ~50ms (model encode + DB insert)

        Args:
            query_text: Raw user query string.

        Returns:
            numpy array of shape (384,)
        """
        # Normalize query for better cache hit rate
        normalized = query_text.strip().lower()

        # 1. Try cache first
        self.cursor.execute(
            """SELECT embedding FROM query_cache WHERE query_text = %s""",
            (normalized,)
        )
        result = self.cursor.fetchone()

        if result:
            # ── Cache HIT ──
            self.cache_stats['hits'] += 1

            # Update hit count and last accessed timestamp
            self.cursor.execute("""
                UPDATE query_cache
                SET hit_count = hit_count + 1,
                    last_accessed = NOW()
                WHERE query_text = %s
            """, (normalized,))
            self.conn.commit()

            return np.array(result[0])

        else:
            # ── Cache MISS ──
            self.cache_stats['misses'] += 1
            embedding = self.model.encode([query_text])[0]

            # Store in cache (ON CONFLICT handles race conditions)
            self.cursor.execute("""
                INSERT INTO query_cache (query_text, embedding)
                VALUES (%s, %s)
                ON CONFLICT (query_text) DO NOTHING
            """, (normalized, embedding.tolist()))
            self.conn.commit()

            return embedding

    # ── Genre Detection ────────────────────────────────────────────────

    def detect_genre(self, query_text):
        """
        Detect genre from query keywords to route to genre-specific HNSW index.

        Genre-specific indexes search ~1,200 movies instead of 11,937,
        yielding ~10x speedup for genre-scoped queries.

        Args:
            query_text: Raw user query string.

        Returns:
            Genre string (e.g., 'Science Fiction') or None.
        """
        query_lower = query_text.lower()

        for genre, keywords in self.GENRE_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return genre

        return None

    # ── Main Search ────────────────────────────────────────────────────

    def search_movies(self, query_text, top_k=5, use_genre_filter=True):
            """Search with caching and optional genre filtering"""
            import time
            start = time.time()
            
            # Get embedding (from cache or compute)
            query_embedding = self.get_query_embedding(query_text)
            
            # Detect genre for optimization
            genre = self.detect_genre(query_text) if use_genre_filter else None
            
            if genre:
                # Use genre filter
                sql = """
                SELECT "movieId", "title", "year", "genres", "avg_rating", "num_ratings",
                    "embedding" <=> %s::vector AS distance
                FROM movies
                WHERE "tmdb_genres" LIKE %s
                ORDER BY "embedding" <=> %s::vector
                LIMIT %s
                """
                self.cursor.execute(sql, (
                    query_embedding.tolist(), 
                    f'%{genre}%', 
                    query_embedding.tolist(), 
                    top_k
                ))
            else:
                # Full search
                sql = """
                SELECT "movieId", "title", "year", "genres", "avg_rating", "num_ratings",
                    "embedding" <=> %s::vector AS distance
                FROM movies
                ORDER BY "embedding" <=> %s::vector
                LIMIT %s
                """
                self.cursor.execute(sql, (
                    query_embedding.tolist(), 
                    query_embedding.tolist(), 
                    top_k
                ))
            
            results = self.cursor.fetchall()
            elapsed = time.time() - start
            
            # Print cache stats
            total = self.cache_stats['hits'] + self.cache_stats['misses']
            if total > 0:
                hit_rate = self.cache_stats['hits'] / total * 100
                print(f"Cache: {hit_rate:.1f}% hit rate ({self.cache_stats['hits']}/{total})")
            
            print(f"Search time: {elapsed*1000:.1f}ms (genre: {genre or 'all'})")
            
            return results
   

    # ── Cache Analytics ────────────────────────────────────────────────

    def get_cache_analytics(self):
        """
        Retrieve persistent cache analytics for course report.

        Returns:
            Dict with 'stats' (total, hits, avg, max) and 'top_queries' list.
        """
        self.cursor.execute("""
            SELECT
                COUNT(*) AS total_queries,
                COALESCE(SUM(hit_count), 0) AS total_hits,
                COALESCE(AVG(hit_count), 0) AS avg_hits,
                COALESCE(MAX(hit_count), 0) AS max_hits
            FROM query_cache
        """)
        stats = self.cursor.fetchone()

        self.cursor.execute("""
            SELECT query_text, hit_count, last_accessed
            FROM query_cache
            ORDER BY hit_count DESC
            LIMIT 10
        """)
        top_queries = self.cursor.fetchall()

        return {
            'stats': {
                'total_cached': stats[0],
                'total_hits': stats[1],
                'avg_hits': float(stats[2]),
                'max_hits': stats[3],
            },
            'top_queries': top_queries,
        }

    def clear_cache(self):
        """Clear the query cache (useful for benchmarking)."""
        self.cursor.execute("TRUNCATE query_cache")
        self.conn.commit()
        self.cache_stats = {'hits': 0, 'misses': 0}
        print("Query cache cleared.")

    # ── Cleanup ────────────────────────────────────────────────────────

    def close(self):
        """Close database connection."""
        self.cursor.close()
        self.conn.close()
        print("Database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ── CLI Interface ──────────────────────────────────────────────────────

def print_results(results, query):
    """Pretty-print search results."""
    print(f"\n🎬 Results for: \"{query}\"")
    print("-" * 60)

    if not results:
        print("  No results found.")
        return

    for rank, (movie_id, title, year, genres, rating, num_ratings, distance) in enumerate(results, 1):
        similarity = 1 - distance  # cosine distance → similarity
        print(f"  {rank}. {title} ({year})")
        print(f"     Genres: {genres}")
        print(f"     Rating: {rating:.1f}⭐ ({num_ratings:,} ratings) | "
              f"Similarity: {similarity:.3f}")


def interactive_search(engine):
    """Run an interactive search loop."""
    print("\n" + "=" * 60)
    print("  VibeLens - Semantic Movie Search")
    print("  Type a vibe, get recommendations!")
    print("  Commands: /stats, /clear, /quit")
    print("=" * 60)

    while True:
        try:
            query = input("\n🔍 Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue

        if query == '/quit':
            break
        elif query == '/stats':
            analytics = engine.get_cache_analytics()
            s = analytics['stats']
            print(f"\n📊 Cache Analytics:")
            print(f"  Cached queries: {s['total_cached']}")
            print(f"  Total hits: {s['total_hits']}")
            print(f"  Avg hits/query: {s['avg_hits']:.1f}")
            print(f"  Max hits: {s['max_hits']}")
            if analytics['top_queries']:
                print(f"\n  Top Queries:")
                for qt, hits, last in analytics['top_queries']:
                    print(f"    {hits:3d} hits — {qt}")
            continue
        elif query == '/clear':
            engine.clear_cache()
            continue

        results = engine.search_movies(query, top_k=5)
        print_results(results, query)


if __name__ == '__main__':
    db_config = {
        'host': os.getenv('PG_HOST'),
        'port': os.getenv('PG_PORT', '5432'),
        'database': os.getenv('PG_DATABASE', 'vibelens'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.getenv('PG_PASSWORD'),
    }

    with MovieSearchEngine(db_config) as engine:
        interactive_search(engine)
