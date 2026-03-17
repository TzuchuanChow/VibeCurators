#!/usr/bin/env python3
"""
VibeLens — End-to-End Pipeline Performance Benchmark
=====================================================

  Phase 1 — Data Ingestion        (Parquet loading + Metadata parsing)
  Phase 2 — Feature Engineering   (Sentence-Transformer vectorization)
  Phase 3 — Vector Search         (pgvector HNSW + Cache hit/miss comparison)
  Phase 4 — Result Generation     (Formatted output, project-specific logic)

Usage:
    python overall_latency.py

Requirements (.env):
    PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
"""

import os
import sys
import time
import json
import statistics
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# -- Add project root to path to import search_movies -------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_movies import MovieSearchEngine

# -- Optional Spark -----------------------------------------------------------
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark not found — Phase 1 will use pandas fallback")

load_dotenv()

# ==============================================================================
#  CONFIG
# ==============================================================================

DB_CONFIG = {
    'host':     os.getenv('PG_HOST'),
    'port':     os.getenv('PG_PORT', '5432'),
    'database': os.getenv('PG_DATABASE', 'vibelens'),
    'user':     os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD'),
}

# Local Parquet path (Used in Phase 1 & 2)
PARQUET_PATH = os.getenv(
    'PARQUET_PATH',
    '../data/preprocessed/movies_with_embeddings.parquet'
)

MODEL_NAME = 'all-MiniLM-L6-v2'

# Test queries for Phase 3 & 4 (covers multiple genres to trigger routing)
BENCHMARK_QUERIES = [
    "dark sci-fi movie about AI and loneliness",
    "funny romantic comedy with happy ending",
    "action movie with intense car chases",
    "animated family film with talking animals",
    "psychological thriller with unexpected twist",
    "emotional drama about grief and loss",
    "scary supernatural horror in an old house",
    "epic fantasy adventure with magic and dragons",
    "true crime documentary style thriller",
    "coming-of-age story set in high school",
]

# ==============================================================================
#  HELPERS
# ==============================================================================

class Timer:
    """Context manager timer that stores elapsed seconds."""
    def __init__(self, label: str = ""):
        self.label   = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def separator(title: str = ""):
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'═' * pad} {title} {'═' * (width - len(title) - 2 - pad)}")
    else:
        print("═" * width)


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


# ==============================================================================
#  PHASE 1 — Data Ingestion  (Parquet → Spark/pandas)
# ==============================================================================

def phase1_data_ingestion() -> dict:
    separator("Phase 1 · Data Ingestion  (Parquet Loading)")
    results = {}

    if not os.path.exists(PARQUET_PATH):
        print(f"\n  ⚠️  Parquet file not found: {PARQUET_PATH}")
        print(f"      Skipping Phase 1, please check the path and retry.")
        return {'phase1_skipped': True, 'phase1_total_sec': 0}

    file_size_mb = os.path.getsize(PARQUET_PATH) / (1024 ** 2)
    print(f"\n  File Size: {file_size_mb:.1f} MB  →  {PARQUET_PATH}")

    # -- 1a. Spark or pandas loading ------------------------------------------
    if SPARK_AVAILABLE:
        print("\n  [1a] Loading Parquet using Spark ...")
        with Timer("spark_load") as t_load:
            spark = (SparkSession.builder
                     .appName("VibeLens-Benchmark")
                     .config("spark.driver.memory", "4g")
                     .config("spark.sql.shuffle.partitions", "8")
                     .getOrCreate())
            spark.sparkContext.setLogLevel("ERROR")
            spark_df  = spark.read.parquet(PARQUET_PATH)
            row_count = spark_df.count()
        results['load_method']  = 'spark'
        results['load_sec']     = t_load.elapsed
        results['row_count']    = row_count
        print(f"      ✓ Spark loaded {row_count:,} rows")
        print(f"      ⏱  Spark Load : {ms(t_load.elapsed)}")
    else:
        print("\n  [1a] Loading Parquet using pandas ...")
        with Timer("pandas_load") as t_load:
            pdf       = pd.read_parquet(PARQUET_PATH)
            row_count = len(pdf)
        results['load_method'] = 'pandas'
        results['load_sec']    = t_load.elapsed
        results['row_count']   = row_count
        print(f"      ✓ pandas loaded {row_count:,} rows")
        print(f"      ⏱  Pandas Load : {ms(t_load.elapsed)}")

    # -- 1b. Metadata parsing (reading specific columns) ----------------------
    print("\n  [1b] Parsing metadata fields ...")
    with Timer("metadata_parse") as t_meta:
        meta_df = pd.read_parquet(
            PARQUET_PATH,
            columns=['movieId', 'title', 'year', 'tmdb_genres',
                     'num_ratings', 'avg_rating']
        )
    results['metadata_parse_sec'] = t_meta.elapsed
    results['file_size_mb']       = round(file_size_mb, 2)
    print(f"      ✓ Parsed {len(meta_df.columns)} columns, {len(meta_df):,} rows")
    print(f"      ⏱  Metadata Parse : {ms(t_meta.elapsed)}")

    results['phase1_total_sec'] = results['load_sec'] + results['metadata_parse_sec']
    print(f"\n  ✅ Phase 1 Total : {ms(results['phase1_total_sec'])}")
    return results


# ==============================================================================
#  PHASE 2 — Feature Engineering  (Sentence-Transformer Vectorization)
# ==============================================================================

def phase2_feature_engineering() -> dict:
    separator("Phase 2 · Feature Engineering  (Sentence-Transformer)")
    results = {}

    from sentence_transformers import SentenceTransformer

    # -- 2a. Model Loading ----------------------------------------------------
    print(f"\n  [2a] Loading model: {MODEL_NAME} ...")
    with Timer("model_load") as t_model:
        model = SentenceTransformer(MODEL_NAME)
    results['model_load_sec'] = t_model.elapsed
    print(f"      ✓ Model loaded (dim=384)")
    print(f"      ⏱  Model Load : {ms(t_model.elapsed)}")

    # -- 2b. Cold start single encoding ---------------------------------------
    print("\n  [2b] Single Query Vectorization (Cold Start) ...")
    with Timer("cold_encode") as t_cold:
        _ = model.encode(["dark sci-fi about AI and loneliness"])
    results['cold_encode_sec'] = t_cold.elapsed
    print(f"      ⏱  Cold Encode : {ms(t_cold.elapsed)}")

    # -- 2c. Warm batch encoding (simulating actual usage) --------------------
    print("\n  [2c] Warm encoding Benchmark Queries (x10) ...")
    warm_times = []
    for q in BENCHMARK_QUERIES:
        with Timer() as t:
            model.encode([q])
        warm_times.append(t.elapsed)
    results['warm_encode_avg_sec'] = statistics.mean(warm_times)
    results['warm_encode_min_sec'] = min(warm_times)
    results['warm_encode_max_sec'] = max(warm_times)
    print(f"      ⏱  Warm Avg     : {ms(results['warm_encode_avg_sec'])}")
    print(f"         Min / Max   : {ms(results['warm_encode_min_sec'])} / "
          f"{ms(results['warm_encode_max_sec'])}")

    # -- 2d. Batch encoding throughput test -----------------------------------
    print("\n  [2d] Batch encoding throughput test ...")
    sample_texts = [f"sample movie soup description number {i}" for i in range(128)]
    batch_results = {}
    for bs in [32, 64, 128]:
        with Timer() as t_batch:
            model.encode(sample_texts[:bs], batch_size=bs)
        tps = bs / t_batch.elapsed
        batch_results[bs] = {'sec': round(t_batch.elapsed, 4),
                             'texts_per_sec': round(tps, 1)}
        print(f"      batch={bs:3d}  →  {ms(t_batch.elapsed):>10}  "
              f"({tps:.0f} texts/sec)")
    results['batch_encoding'] = batch_results

    results['phase2_total_sec'] = (
        results['model_load_sec'] + results['cold_encode_sec']
    )
    print(f"\n  ✅ Phase 2 Total : {ms(results['phase2_total_sec'])}")
    return results, model


# ==============================================================================
#  PHASE 3 — Vector Search  (pgvector HNSW + Cache)
# ==============================================================================

def phase3_vector_search(model) -> dict:
    separator("Phase 3 · Vector Search  (pgvector HNSW + Cache)")
    results = {}

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # -- 3a. Clear cache for fair baseline ------------------------------------
    print("\n  [3a] Truncating query_cache for fair test ...")
    cursor.execute("TRUNCATE query_cache")
    conn.commit()
    print("      ✓ Cache cleared")

    # -- 3b. Cache MISS latency (10 new queries) ------------------------------
    print("\n  [3b] Cache MISS Latency (10 new queries) ...")
    miss_times   = []
    embed_times  = []
    search_times = []

    for query in BENCHMARK_QUERIES:
        normalized = query.strip().lower()

        with Timer() as t_embed:
            embedding = model.encode([query])[0]
        embed_times.append(t_embed.elapsed)

        with Timer() as t_search:
            cursor.execute("""
                SELECT "movieId", "title", "year", "tmdb_genres",
                       "avg_rating", "num_ratings",
                       "embedding" <=> %s::vector AS distance
                FROM movies
                ORDER BY "embedding" <=> %s::vector
                LIMIT 5
            """, (embedding.tolist(), embedding.tolist()))
            cursor.fetchall()
        search_times.append(t_search.elapsed)

        # Write to cache
        cursor.execute("""
            INSERT INTO query_cache (query_text, embedding)
            VALUES (%s, %s)
            ON CONFLICT (query_text) DO NOTHING
        """, (normalized, embedding.tolist()))
        conn.commit()

        miss_times.append(t_embed.elapsed + t_search.elapsed)

    results['cache_miss'] = {
        'avg_total_sec':  statistics.mean(miss_times),
        'avg_embed_sec':  statistics.mean(embed_times),
        'avg_search_sec': statistics.mean(search_times),
        'min_total_sec':  min(miss_times),
        'max_total_sec':  max(miss_times),
    }
    print(f"      ⏱  Avg Total   : {ms(results['cache_miss']['avg_total_sec'])}")
    print(f"         ├ Embed     : {ms(results['cache_miss']['avg_embed_sec'])}")
    print(f"         └ DB Search : {ms(results['cache_miss']['avg_search_sec'])}")

    # -- 3c. Cache HIT latency (repeating same queries) -----------------------
    print("\n  [3c] Cache HIT Latency (repeating same 10 queries) ...")
    hit_times = []

    for query in BENCHMARK_QUERIES:
        normalized = query.strip().lower()
        with Timer() as t_hit:
            # Read embedding from cache
            cursor.execute(
                "SELECT embedding FROM query_cache WHERE query_text = %s",
                (normalized,)
            )
            cached_emb = np.array(cursor.fetchone()[0])

            # Update hit statistics
            cursor.execute("""
                UPDATE query_cache
                SET hit_count = hit_count + 1, last_accessed = NOW()
                WHERE query_text = %s
            """, (normalized,))
            conn.commit()

            # Vector search
            cursor.execute("""
                SELECT "movieId", "title", "year", "tmdb_genres",
                       "avg_rating", "num_ratings",
                       "embedding" <=> %s::vector AS distance
                FROM movies
                ORDER BY "embedding" <=> %s::vector
                LIMIT 5
            """, (cached_emb.tolist(), cached_emb.tolist()))
            cursor.fetchall()
        hit_times.append(t_hit.elapsed)

    results['cache_hit'] = {
        'avg_total_sec': statistics.mean(hit_times),
        'min_total_sec': min(hit_times),
        'max_total_sec': max(hit_times),
    }
    print(f"      ⏱  Avg Total (HIT) : {ms(results['cache_hit']['avg_total_sec'])}")
    print(f"         Min / Max       : "
          f"{ms(results['cache_hit']['min_total_sec'])} / "
          f"{ms(results['cache_hit']['max_total_sec'])}")

    # -- 3d. Genre filter vs Full scan comparison -----------------------------
    print("\n  [3d] Genre Filter vs Full Scan Comparison ...")
    test_query = "dark sci-fi movie about AI"
    test_emb   = model.encode([test_query])[0].tolist()

    with Timer() as t_full:
        cursor.execute("""
            SELECT "movieId", "title"
            FROM movies
            ORDER BY "embedding" <=> %s::vector
            LIMIT 5
        """, (test_emb,))
        cursor.fetchall()

    with Timer() as t_genre:
        cursor.execute("""
            SELECT "movieId", "title"
            FROM movies
            WHERE "tmdb_genres" LIKE %s
            ORDER BY "embedding" <=> %s::vector
            LIMIT 5
        """, ('%Science Fiction%', test_emb))
        cursor.fetchall()

    results['full_scan_sec']      = t_full.elapsed
    results['genre_filtered_sec'] = t_genre.elapsed
    speedup = (t_full.elapsed / t_genre.elapsed) if t_genre.elapsed > 0 else 0
    print(f"      ⏱  Full Scan    : {ms(t_full.elapsed)}")
    print(f"      ⏱  Genre Filter : {ms(t_genre.elapsed)}")
    print(f"      🚀 Speedup      : {speedup:.1f}×")

    # -- 3e. Cache analytics --------------------------------------------------
    cursor.execute("""
        SELECT COUNT(*), COALESCE(SUM(hit_count), 0), COALESCE(AVG(hit_count), 0)
        FROM query_cache
    """)
    total_cached, total_hits, avg_hits = cursor.fetchone()
    n        = len(BENCHMARK_QUERIES)
    hit_rate = n / (n * 2) * 100
    results['cache_analytics']      = {
        'total_cached': total_cached,
        'total_hits':   int(total_hits),
        'avg_hits':     float(avg_hits),
    }
    results['session_hit_rate_pct'] = hit_rate
    print(f"\n      📊 Cached queries    : {total_cached}")
    print(f"         Total hits       : {total_hits}")
    print(f"         Session hit rate : {hit_rate:.0f}%")

    cursor.close()
    conn.close()

    results['phase3_total_sec'] = (
        results['cache_miss']['avg_total_sec'] * n +
        results['cache_hit']['avg_total_sec']  * n
    )
    print(f"\n  ✅ Phase 3 Avg Cache MISS : {ms(results['cache_miss']['avg_total_sec'])}")
    print(f"     Phase 3 Avg Cache HIT  : {ms(results['cache_hit']['avg_total_sec'])}")
    return results


# ==============================================================================
#  PHASE 4 — Result Generation  (Native search_movies.py, no API key)
# ==============================================================================

def phase4_result_generation() -> dict:
    """
    Uses MovieSearchEngine to run the full end-to-end pipeline:
      embed query → cache lookup/miss → pgvector search → formatted output

    No external LLM APIs are called; tests local project performance.
    """
    separator("Phase 4 · Result Generation  (MovieSearchEngine E2E)")
    results = {}

    rag_queries = BENCHMARK_QUERIES[:5]   # Use first 5 for full chain test

    with MovieSearchEngine(DB_CONFIG, model_name=MODEL_NAME) as engine:

        # -- 4a. Clear cache to ensure first round are MISSes ------------------
        engine.clear_cache()

        # -- 4b. Round 1: Cache MISS (Cold Start) -----------------------------
        print("\n  [4a] Round 1 — Cache MISS (Cold Start, 5 queries) ...")
        miss_timings = []

        for query in rag_queries:
            with Timer() as t_total:
                # embed + cache lookup + pgvector search
                search_results = engine.search_movies(query, top_k=5)

                # Formatted output (native result display logic)
                output_lines = []
                for rank, (mid, title, year, genres, rating, nrat, dist) in \
                        enumerate(search_results, 1):
                    sim = 1 - dist
                    output_lines.append(
                        f"{rank}. {title} ({year}) | {genres} | "
                        f"{rating:.1f}⭐ ({nrat:,} ratings) | sim={sim:.3f}"
                    )
                formatted_output = "\n".join(output_lines)

            miss_timings.append(t_total.elapsed)
            print(f"      Query: \"{query[:45]}...\"")
            print(f"      ⏱  Total : {ms(t_total.elapsed)}  "
                  f"| Results: {len(search_results)}")

        # -- 4c. Round 2: Cache HIT (Repeating queries) -----------------------
        print("\n  [4b] Round 2 — Cache HIT (Repeated 5 queries) ...")
        hit_timings = []

        for query in rag_queries:
            with Timer() as t_total:
                search_results = engine.search_movies(query, top_k=5)
                output_lines = []
                for rank, (mid, title, year, genres, rating, nrat, dist) in \
                        enumerate(search_results, 1):
                    sim = 1 - dist
                    output_lines.append(
                        f"{rank}. {title} ({year}) | {genres} | "
                        f"{rating:.1f}⭐ ({nrat:,} ratings) | sim={sim:.3f}"
                    )
                formatted_output = "\n".join(output_lines)

            hit_timings.append(t_total.elapsed)
            print(f"      Query: \"{query[:45]}...\"")
            print(f"      ⏱  Total : {ms(t_total.elapsed)}")

        # -- Stats ------------------------------------------------------------
        results['cache_miss'] = {
            'avg_total_sec': statistics.mean(miss_timings),
            'min_total_sec': min(miss_timings),
            'max_total_sec': max(miss_timings),
        }
        results['cache_hit'] = {
            'avg_total_sec': statistics.mean(hit_timings),
            'min_total_sec': min(hit_timings),
            'max_total_sec': max(hit_timings),
        }

        # Session cache stats
        session = engine.cache_stats
        total_req = session['hits'] + session['misses']
        hit_rate  = session['hits'] / total_req * 100 if total_req else 0
        results['session_cache_stats'] = {
            'hits':     session['hits'],
            'misses':   session['misses'],
            'hit_rate': round(hit_rate, 1),
        }

        # Persistent cache analytics
        analytics = engine.get_cache_analytics()
        results['db_cache_analytics'] = analytics['stats']

    print(f"\n  ✅ Phase 4 Avg (MISS) : {ms(results['cache_miss']['avg_total_sec'])}")
    print(f"     Phase 4 Avg (HIT)  : {ms(results['cache_hit']['avg_total_sec'])}")
    print(f"     Session hit rate   : {hit_rate:.0f}%")
    return results


# ==============================================================================
#  SUMMARY REPORT
# ==============================================================================

def print_summary(p1, p2, p3, p4):
    separator("END-TO-END PERFORMANCE SUMMARY")

    col_w = [18, 26, 40, 14]
    header = (f"  {'Phase':<{col_w[0]}}{'Description':<{col_w[1]}}"
              f"{'Details':<{col_w[2]}}{'Latency':>{col_w[3]}}")
    print(header)
    print(f"  {'─' * sum(col_w)}")

    rows = []

    # Phase 1
    if not p1.get('phase1_skipped'):
        rows.append(("Phase 1", "Data Ingestion",
                     f"{p1.get('load_method','?').upper()} Load + Meta Parse",
                     p1['phase1_total_sec']))

    # Phase 2
    rows.append(("Phase 2", "Feature Eng.",
                 "Model Load + Cold Vectorization",
                 p2['phase2_total_sec']))

    # Phase 3
    rows.append(("Phase 3 (miss)", "Vector Search (Cold)",
                 "Embed + pgvector HNSW",
                 p3['cache_miss']['avg_total_sec']))
    rows.append(("Phase 3 (hit)", "Vector Search (Hit)",
                 "Cache Lookup + pgvector HNSW",
                 p3['cache_hit']['avg_total_sec']))

    # Phase 4
    rows.append(("Phase 4 (miss)", "E2E Generation (Cold)",
                 "MovieSearchEngine Full Pipeline",
                 p4['cache_miss']['avg_total_sec']))
    rows.append(("Phase 4 (hit)", "E2E Generation (Hit)",
                 "Cache Hit Full Pipeline",
                 p4['cache_hit']['avg_total_sec']))

    for phase, desc, detail, secs in rows:
        print(f"  {phase:<{col_w[0]}}{desc:<{col_w[1]}}"
              f"{detail:<{col_w[2]}}{ms(secs):>{col_w[3]}}")

    print(f"\n  {'─' * sum(col_w)}")

    # Sub-breakdown
    print(f"\n  Phase 3 Breakdown:")
    print(f"    {'Embed (sentence-transformer)':<42} "
          f"{ms(p3['cache_miss']['avg_embed_sec'])}")
    print(f"    {'DB Search (pgvector cosine)':<42} "
          f"{ms(p3['cache_miss']['avg_search_sec'])}")
    speedup = (p3['full_scan_sec'] / p3['genre_filtered_sec']
               if p3['genre_filtered_sec'] > 0 else 0)
    print(f"    {'Genre filter speedup ratio':<42} {speedup:.1f}×")
    print(f"    {'Session cache hit rate':<42} "
          f"{p3['session_hit_rate_pct']:.0f}%")

    print(f"\n  Phase 4 Breakdown (MovieSearchEngine):")
    print(f"    {'Avg Cache MISS Latency':<42} "
          f"{ms(p4['cache_miss']['avg_total_sec'])}")
    print(f"    {'Avg Cache HIT Latency':<42} "
          f"{ms(p4['cache_hit']['avg_total_sec'])}")
    print(f"    {'Cache Hit Rate (session)':<42} "
          f"{p4['session_cache_stats']['hit_rate']:.0f}%")
    speedup_e2e = (p4['cache_miss']['avg_total_sec'] / p4['cache_hit']['avg_total_sec']
                   if p4['cache_hit']['avg_total_sec'] > 0 else 0)
    print(f"    {'Speedup Ratio (HIT vs MISS)':<42} "
          f"{speedup_e2e:.1f}×" if speedup_e2e else "    N/A")

    # User Perceived Latency
    print(f"\n  {'─' * sum(col_w)}")
    print(f"  {'User Perceived Latency (Cold Start)':<{col_w[0]+col_w[1]+col_w[2]}} "
          f"{ms(p4['cache_miss']['avg_total_sec']):>{col_w[3]}}")
    print(f"  {'User Perceived Latency (Cache Hit)':<{col_w[0]+col_w[1]+col_w[2]}} "
          f"{ms(p4['cache_hit']['avg_total_sec']):>{col_w[3]}}")

    separator()

    # -- Save JSON Report -----------------------------------------------------
    report = {
        'phase1': p1,
        'phase2': {k: v for k, v in p2.items() if k != '__model__'},
        'phase3': p3,
        'phase4': p4,
        'summary': {
            'user_latency_cold_ms':   round(p4['cache_miss']['avg_total_sec'] * 1000, 1),
            'user_latency_cached_ms': round(p4['cache_hit']['avg_total_sec']  * 1000, 1),
        }
    }
    report_path = "overall_latency_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  📄 Full report saved → {report_path}")


# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == '__main__':
    separator("VibeLens — Pipeline Performance Benchmark")
    print(f"  Queries  : {len(BENCHMARK_QUERIES)}")
    print(f"  Model    : {MODEL_NAME}")
    print(f"  DB host  : {DB_CONFIG['host']}")
    print(f"  Parquet  : {PARQUET_PATH}")

    p1          = phase1_data_ingestion()
    p2, model   = phase2_feature_engineering()
    p3          = phase3_vector_search(model)
    p4          = phase4_result_generation()

    print_summary(p1, p2, p3, p4)