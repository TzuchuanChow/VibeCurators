#!/usr/bin/env python3
"""
Test query caching and genre-specific index performance.
Run after load_to_postgres.py has created the cache table and genre indexes.
"""

import time
from search_movies import MovieSearchEngine, print_results
from dotenv import load_dotenv
import os

load_dotenv()

db_config = {
    'host': os.getenv('PG_HOST'),
    'port': os.getenv('PG_PORT', '5432'),
    'database': os.getenv('PG_DATABASE', 'vibelens'),
    'user': os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD'),
}

# Test queries — includes repeats to demonstrate cache hits
test_queries = [
    "dark sci-fi movie about AI",           # Miss → Sci-Fi index
    "funny romantic comedy",                 # Miss → Comedy index
    "action movie with car chases",          # Miss → Action index
    "dark sci-fi movie about AI",            # HIT  (repeat)
    "animated family movie",                 # Miss → Animation index
    "dark sci-fi movie about AI",            # HIT  (repeat)
    "emotional drama about loss",            # Miss → Drama index
    "scary monster horror film",             # Miss → Horror index
    "funny romantic comedy",                 # HIT  (repeat)
    "a mind-bending thriller with twists",   # Miss → Thriller index
]

print("=" * 60)
print("  VibeLens — Cache & Genre Index Test Suite")
print("=" * 60)

with MovieSearchEngine(db_config) as engine:
    # Clear cache for clean benchmark
    engine.clear_cache()

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: \"{query}\"")
        results = engine.search_movies(query, top_k=3)

        for rank, (mid, title, year, genres, rating, nrat, dist) in enumerate(results, 1):
            sim = 1 - dist
            print(f"  {rank}. {title} ({year}) — {rating:.1f}⭐ "
                  f"({nrat:,} ratings) [sim={sim:.3f}]")

    # ── Cache Analytics ──
    print(f"\n{'='*60}")
    print("  Cache Analytics")
    print(f"{'='*60}")

    analytics = engine.get_cache_analytics()
    s = analytics['stats']

    print(f"  Total cached queries:  {s['total_cached']}")
    print(f"  Total cache hits:      {s['total_hits']}")
    print(f"  Avg hits per query:    {s['avg_hits']:.1f}")
    print(f"  Max hits (single):     {s['max_hits']}")

    session = engine.cache_stats
    total = session['hits'] + session['misses']
    hit_rate = session['hits'] / total * 100 if total else 0

    print(f"\n  Session stats:")
    print(f"    Queries run:  {total}")
    print(f"    Cache hits:   {session['hits']}")
    print(f"    Cache misses: {session['misses']}")
    print(f"    Hit rate:     {hit_rate:.1f}%")

    if analytics['top_queries']:
        print(f"\n  Top Queries:")
        for qt, hits, last in analytics['top_queries']:
            print(f"    {hits:3d} hits — \"{qt}\"")

print("\n✅ Test complete.")
