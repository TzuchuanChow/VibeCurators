#!/usr/bin/env python3
"""
VibeLens — End-to-End Pipeline Performance Benchmark
=====================================================
Measures and reports timing for all four phases:
  Phase 1 — Data Ingestion      (S3 → Spark)
  Phase 2 — Feature Engineering (Vectorization + Metadata)
  Phase 3 — Vector Search        (pgvector HNSW Retrieval)
  Phase 4 — RAG Generation       (LLM Inference via Anthropic)

Usage:
    python benchmark_pipeline.py

Requirements (.env):
    PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET, S3_PARQUET_KEY
    ANTHROPIC_API_KEY  (for RAG phase)
"""

import os
import time
import json
import statistics
import psycopg2
import numpy as np
import anthropic
import boto3
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ── Optional Spark import (graceful fallback if not installed) ─────────
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  PySpark not found — Phase 1 will use boto3 + pandas fallback.")

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

DB_CONFIG = {
    'host':     os.getenv('PG_HOST'),
    'port':     os.getenv('PG_PORT', '5432'),
    'database': os.getenv('PG_DATABASE', 'vibelens'),
    'user':     os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD'),
}

S3_BUCKET      = os.getenv('S3_BUCKET', 'vibelens-data')
S3_PARQUET_KEY = os.getenv('S3_PARQUET_KEY',
                            'preprocessed/movies_with_embeddings.parquet')
MODEL_NAME     = 'all-MiniLM-L6-v2'

# Queries used for Phase 3 & 4 benchmarks
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

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════

class Timer:
    """Context-manager timer that stores elapsed seconds."""
    def __init__(self, label: str):
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
        print(f"\n{'═' * pad} {title} {'═' * pad}")
    else:
        print("═" * width)


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def fmt_rows(n: int) -> str:
    return f"{n:,}"


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — DATA INGESTION  (S3 → Spark / pandas fallback)
# ══════════════════════════════════════════════════════════════════════

def phase1_data_ingestion() -> dict:
    separator("Phase 1 · Data Ingestion  (S3 → Spark)")

    results = {}

    # ── 1a. S3 Download ───────────────────────────────────────────────
    print("\n  [1a] Downloading Parquet from S3 …")
    local_path = "/tmp/movies_with_embeddings.parquet"

    with Timer("s3_download") as t_s3:
        try:
            s3 = boto3.client(
                's3',
                aws_access_key_id     = os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name           = os.getenv('AWS_REGION', 'us-east-1'),
            )
            s3.download_file(S3_BUCKET, S3_PARQUET_KEY, local_path)
            file_size_mb = os.path.getsize(local_path) / (1024 ** 2)
            print(f"      ✓ Downloaded {file_size_mb:.1f} MB  →  {local_path}")
        except Exception as e:
            print(f"      ⚠️  S3 download skipped ({e})")
            print(f"         Falling back to local file …")
            local_path = "../data/preprocessed/movies_with_embeddings.parquet"
            file_size_mb = (os.path.getsize(local_path) / (1024 ** 2)
                            if os.path.exists(local_path) else 0)

    results['s3_download_sec']  = t_s3.elapsed
    results['file_size_mb']     = round(file_size_mb, 2)
    print(f"      ⏱  S3 Download : {ms(t_s3.elapsed)}")

    # ── 1b. Spark Load ────────────────────────────────────────────────
    print("\n  [1b] Loading Parquet into Spark …")

    if SPARK_AVAILABLE and os.path.exists(local_path):
        with Timer("spark_load") as t_spark:
            spark = (SparkSession.builder
                     .appName("VibeLens-Benchmark")
                     .config("spark.driver.memory", "4g")
                     .config("spark.sql.shuffle.partitions", "8")
                     .getOrCreate())
            spark.sparkContext.setLogLevel("ERROR")
            spark_df   = spark.read.parquet(local_path)
            row_count  = spark_df.count()

        results['spark_load_sec'] = t_spark.elapsed
        results['row_count']      = row_count
        print(f"      ✓ Loaded {fmt_rows(row_count)} rows into Spark")
        print(f"      ⏱  Spark Load   : {ms(t_spark.elapsed)}")
    else:
        print("      ⚠️  Spark unavailable — using pandas …")
        with Timer("pandas_load") as t_pd:
            pdf      = pd.read_parquet(local_path)
            row_count = len(pdf)
        results['spark_load_sec'] = t_pd.elapsed
        results['row_count']      = row_count
        print(f"      ✓ Loaded {fmt_rows(row_count)} rows via pandas")
        print(f"      ⏱  Pandas Load  : {ms(t_pd.elapsed)}")

    # ── 1c. Metadata Parse ────────────────────────────────────────────
    print("\n  [1c] Parsing metadata fields …")
    with Timer("metadata_parse") as t_meta:
        pdf_sample = pd.read_parquet(local_path,
                                     columns=['movieId', 'title', 'year',
                                              'tmdb_genres', 'num_ratings',
                                              'avg_rating'])
    results['metadata_parse_sec'] = t_meta.elapsed
    print(f"      ✓ Parsed {len(pdf_sample.columns)} metadata columns")
    print(f"      ⏱  Metadata Parse : {ms(t_meta.elapsed)}")

    results['phase1_total_sec'] = (
        results['s3_download_sec'] +
        results['spark_load_sec'] +
        results['metadata_parse_sec']
    )
    print(f"\n  ✅ Phase 1 Total : {ms(results['phase1_total_sec'])}")
    return results


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2 — FEATURE ENGINEERING  (Vectorization + Metadata)
# ══════════════════════════════════════════════════════════════════════

def phase2_feature_engineering() -> dict:
    separator("Phase 2 · Feature Engineering  (Vectorization)")

    results = {}

    # ── 2a. Model Load ────────────────────────────────────────────────
    print(f"\n  [2a] Loading Sentence-Transformer model: {MODEL_NAME} …")
    with Timer("model_load") as t_model:
        model = SentenceTransformer(MODEL_NAME)
    results['model_load_sec'] = t_model.elapsed
    print(f"      ✓ Model loaded  (dim=384)")
    print(f"      ⏱  Model Load   : {ms(t_model.elapsed)}")

    # ── 2b. Single-query encode ───────────────────────────────────────
    print("\n  [2b] Single-query vectorization (cold) …")
    with Timer("single_encode_cold") as t_cold:
        _ = model.encode(["dark sci-fi about AI and loneliness"])
    results['single_encode_cold_sec'] = t_cold.elapsed
    print(f"      ⏱  Cold Encode  : {ms(t_cold.elapsed)}")

    print("  [2b] Single-query vectorization (warm, ×10) …")
    warm_times = []
    for q in BENCHMARK_QUERIES:
        with Timer("") as t:
            model.encode([q])
        warm_times.append(t.elapsed)
    results['single_encode_warm_avg_sec'] = statistics.mean(warm_times)
    results['single_encode_warm_min_sec'] = min(warm_times)
    results['single_encode_warm_max_sec'] = max(warm_times)
    print(f"      ⏱  Warm Avg     : {ms(results['single_encode_warm_avg_sec'])}")
    print(f"      ⏱  Warm Min/Max : {ms(results['single_encode_warm_min_sec'])} / "
          f"{ms(results['single_encode_warm_max_sec'])}")

    # ── 2c. Batch encode (simulates generation pipeline) ──────────────
    BATCH_SIZES = [32, 64, 128]
    sample_texts = [f"sample movie description number {i}" for i in range(128)]
    print("\n  [2c] Batch encoding throughput …")
    batch_results = {}
    for bs in BATCH_SIZES:
        with Timer("") as t_batch:
            model.encode(sample_texts[:bs], batch_size=bs)
        tps = bs / t_batch.elapsed
        batch_results[bs] = {'sec': t_batch.elapsed, 'texts_per_sec': round(tps, 1)}
        print(f"      batch={bs:3d}  →  {ms(t_batch.elapsed):>10}  "
              f"({tps:.0f} texts/sec)")
    results['batch_encoding'] = batch_results

    # ── 2d. TF-IDF metadata processing (Spark ML) ────────────────────
    print("\n  [2d] TF-IDF metadata processing …")
    if SPARK_AVAILABLE:
        try:
            from pyspark.sql import SparkSession
            from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
            from pyspark.ml import Pipeline

            spark = (SparkSession.builder.appName("VibeLens-Benchmark")
                     .config("spark.driver.memory", "4g").getOrCreate())
            spark.sparkContext.setLogLevel("ERROR")

            local_path = "/tmp/movies_with_embeddings.parquet"
            if not os.path.exists(local_path):
                local_path = "../data/preprocessed/movies_with_embeddings.parquet"

            movies_df = spark.read.parquet(local_path)

            with Timer("tfidf_fit_transform") as t_tfidf:
                pipeline = Pipeline(stages=[
                    Tokenizer(inputCol="soup", outputCol="words"),
                    StopWordsRemover(inputCol="words", outputCol="filtered_words"),
                    HashingTF(inputCol="filtered_words", outputCol="rawFeatures",
                              numFeatures=10000),
                    IDF(inputCol="rawFeatures", outputCol="features"),
                ])
                tfidf_model = pipeline.fit(movies_df)
                tfidf_df    = tfidf_model.transform(movies_df)
                tfidf_df.count()   # force materialization

            results['tfidf_fit_transform_sec'] = t_tfidf.elapsed
            print(f"      ✓ TF-IDF pipeline fit + transform complete")
            print(f"      ⏱  TF-IDF Pipeline : {ms(t_tfidf.elapsed)}")
        except Exception as e:
            print(f"      ⚠️  TF-IDF skipped: {e}")
            results['tfidf_fit_transform_sec'] = None
    else:
        print("      ⚠️  Spark not available — TF-IDF skipped")
        results['tfidf_fit_transform_sec'] = None

    results['phase2_total_sec'] = (
        results['model_load_sec'] +
        results['single_encode_cold_sec'] +
        (results.get('tfidf_fit_transform_sec') or 0)
    )
    print(f"\n  ✅ Phase 2 Total : {ms(results['phase2_total_sec'])}")
    return results, model


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3 — VECTOR SEARCH  (pgvector HNSW Retrieval)
# ══════════════════════════════════════════════════════════════════════

def phase3_vector_search(model) -> dict:
    separator("Phase 3 · Vector Search  (pgvector HNSW Retrieval)")

    results = {}

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # ── 3a. Cache warm-up: clear for fair benchmark ───────────────────
    print("\n  [3a] Clearing query cache for clean benchmark …")
    cursor.execute("TRUNCATE query_cache")
    conn.commit()
    print("      ✓ Cache cleared")

    # ── 3b. Cache MISS latency (first-time queries) ───────────────────
    print("\n  [3b] Cache MISS latency (10 unique queries) …")
    miss_times   = []
    embed_times  = []
    search_times = []

    for query in BENCHMARK_QUERIES:
        normalized = query.strip().lower()

        # Embedding time
        with Timer("") as t_embed:
            embedding = model.encode([query])[0]
        embed_times.append(t_embed.elapsed)

        # Search time
        with Timer("") as t_search:
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

        # Store in cache
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
    print(f"      ⏱  Avg Total    : {ms(results['cache_miss']['avg_total_sec'])}")
    print(f"         ├ Embed      : {ms(results['cache_miss']['avg_embed_sec'])}")
    print(f"         └ DB Search  : {ms(results['cache_miss']['avg_search_sec'])}")

    # ── 3c. Cache HIT latency (repeat same queries) ───────────────────
    print("\n  [3c] Cache HIT latency (same 10 queries repeated) …")
    hit_times = []

    for query in BENCHMARK_QUERIES:
        normalized = query.strip().lower()
        with Timer("") as t_hit:
            # Lookup from cache
            cursor.execute(
                "SELECT embedding FROM query_cache WHERE query_text = %s",
                (normalized,)
            )
            cached_emb = np.array(cursor.fetchone()[0])

            # Update hit count
            cursor.execute("""
                UPDATE query_cache
                SET hit_count = hit_count + 1, last_accessed = NOW()
                WHERE query_text = %s
            """, (normalized,))
            conn.commit()

            # Vector search with cached embedding
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
    print(f"      ⏱  Avg Total (cache hit) : {ms(results['cache_hit']['avg_total_sec'])}")
    print(f"         Min / Max             : "
          f"{ms(results['cache_hit']['min_total_sec'])} / "
          f"{ms(results['cache_hit']['max_total_sec'])}")

    # ── 3d. Genre-filtered vs full-scan ──────────────────────────────
    print("\n  [3d] Genre-filtered index vs full scan …")
    test_query    = "dark sci-fi movie about AI"
    test_emb      = model.encode([test_query])[0].tolist()

    with Timer("") as t_full:
        cursor.execute("""
            SELECT "movieId", "title"
            FROM movies
            ORDER BY "embedding" <=> %s::vector
            LIMIT 5
        """, (test_emb,))
        cursor.fetchall()

    with Timer("") as t_genre:
        cursor.execute("""
            SELECT "movieId", "title"
            FROM movies
            WHERE "tmdb_genres" LIKE %s
            ORDER BY "embedding" <=> %s::vector
            LIMIT 5
        """, ('%Science Fiction%', test_emb))
        cursor.fetchall()

    results['full_scan_sec']     = t_full.elapsed
    results['genre_filtered_sec'] = t_genre.elapsed
    speedup = t_full.elapsed / t_genre.elapsed if t_genre.elapsed > 0 else 0
    print(f"      ⏱  Full Scan     : {ms(t_full.elapsed)}")
    print(f"      ⏱  Genre Filter  : {ms(t_genre.elapsed)}")
    print(f"      🚀 Speedup       : {speedup:.1f}×")

    # ── 3e. Cache analytics ───────────────────────────────────────────
    cursor.execute("""
        SELECT COUNT(*), COALESCE(SUM(hit_count), 0),
               COALESCE(AVG(hit_count), 0)
        FROM query_cache
    """)
    total_cached, total_hits, avg_hits = cursor.fetchone()
    results['cache_analytics'] = {
        'total_cached': total_cached,
        'total_hits':   int(total_hits),
        'avg_hits':     float(avg_hits),
    }

    # Cache hit rate this session
    n = len(BENCHMARK_QUERIES)
    hit_rate = n / (n * 2) * 100   # second pass was all hits
    results['session_hit_rate_pct'] = hit_rate
    print(f"\n      📊 Cached queries : {total_cached}")
    print(f"         Total hits    : {total_hits}")
    print(f"         Session hit rate: {hit_rate:.0f}%")

    cursor.close()
    conn.close()

    results['phase3_total_sec'] = (
        results['cache_miss']['avg_total_sec'] * n +
        results['cache_hit']['avg_total_sec']  * n
    )
    print(f"\n  ✅ Phase 3 Avg Search Latency:")
    print(f"     Cache MISS → {ms(results['cache_miss']['avg_total_sec'])}")
    print(f"     Cache HIT  → {ms(results['cache_hit']['avg_total_sec'])}")
    return results


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 — RAG GENERATION  (LLM Inference via Anthropic Claude)
# ══════════════════════════════════════════════════════════════════════

def phase4_rag_generation(model) -> dict:
    separator("Phase 4 · RAG Generation  (LLM Inference)")

    results = {}

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    if not ANTHROPIC_API_KEY:
        print("\n  ⚠️  ANTHROPIC_API_KEY not set — Phase 4 skipped.")
        return {'phase4_skipped': True}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── 4a. Retrieval (reuse pgvector) ────────────────────────────────
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    rag_queries = BENCHMARK_QUERIES[:3]   # run 3 full RAG cycles
    rag_timings = []

    for query in rag_queries:
        print(f"\n  Query: \"{query}\"")
        timing = {}

        # Step A — embed query
        with Timer("") as t_embed:
            q_emb = model.encode([query])[0]
        timing['embed_sec'] = t_embed.elapsed

        # Step B — retrieve top-5 from pgvector
        with Timer("") as t_retrieve:
            cursor.execute("""
                SELECT "title", "year", "tmdb_genres", "avg_rating", "soup"
                FROM movies
                ORDER BY "embedding" <=> %s::vector
                LIMIT 5
            """, (q_emb.tolist(),))
            top_movies = cursor.fetchall()
        timing['retrieve_sec'] = t_retrieve.elapsed

        # Build context for LLM
        context_lines = []
        for i, (title, year, genres, rating, soup) in enumerate(top_movies, 1):
            snippet = (soup or "")[:300]
            context_lines.append(
                f"{i}. {title} ({year}) | Genres: {genres} | "
                f"Rating: {rating:.1f}\n   {snippet}"
            )
        context = "\n".join(context_lines)

        prompt = f"""You are a movie recommendation assistant.

A user is looking for: "{query}"

Here are the top 5 retrieved movies:
{context}

For each movie, write ONE sentence explaining why it matches the user's request.
Be concise and specific."""

        # Step C — LLM inference
        with Timer("") as t_llm:
            response = client.messages.create(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 512,
                messages   = [{"role": "user", "content": prompt}],
            )
            llm_output = response.content[0].text

        timing['llm_sec']    = t_llm.elapsed
        timing['total_sec']  = (timing['embed_sec'] +
                                timing['retrieve_sec'] +
                                timing['llm_sec'])
        timing['tokens_out'] = response.usage.output_tokens

        rag_timings.append(timing)

        print(f"      ⏱  Embed     : {ms(timing['embed_sec'])}")
        print(f"      ⏱  Retrieve  : {ms(timing['retrieve_sec'])}")
        print(f"      ⏱  LLM       : {ms(timing['llm_sec'])}  "
              f"({timing['tokens_out']} output tokens)")
        print(f"      ⏱  Total RAG : {ms(timing['total_sec'])}")

    cursor.close()
    conn.close()

    results['per_query']          = rag_timings
    results['avg_embed_sec']      = statistics.mean(t['embed_sec']    for t in rag_timings)
    results['avg_retrieve_sec']   = statistics.mean(t['retrieve_sec'] for t in rag_timings)
    results['avg_llm_sec']        = statistics.mean(t['llm_sec']      for t in rag_timings)
    results['avg_total_rag_sec']  = statistics.mean(t['total_sec']    for t in rag_timings)

    print(f"\n  ✅ Phase 4 Averages (over {len(rag_queries)} queries):")
    print(f"     Embed    : {ms(results['avg_embed_sec'])}")
    print(f"     Retrieve : {ms(results['avg_retrieve_sec'])}")
    print(f"     LLM      : {ms(results['avg_llm_sec'])}")
    print(f"     Total    : {ms(results['avg_total_rag_sec'])}")

    return results


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════

def print_summary(p1, p2, p3, p4):
    separator("END-TO-END PERFORMANCE SUMMARY")

    rows = [
        ("Phase 1", "Data Ingestion",
         f"S3 download + Spark load + metadata parse",
         p1.get('phase1_total_sec', 0)),
        ("Phase 2", "Feature Engineering",
         f"Model load + vectorization + TF-IDF",
         p2.get('phase2_total_sec', 0)),
        ("Phase 3 (miss)", "Vector Search (cold)",
         f"Embed query + pgvector HNSW retrieval",
         p3['cache_miss']['avg_total_sec']),
        ("Phase 3 (hit)", "Vector Search (cached)",
         f"Cache lookup + pgvector HNSW retrieval",
         p3['cache_hit']['avg_total_sec']),
    ]

    if not p4.get('phase4_skipped'):
        rows.append(("Phase 4", "RAG Generation",
                     f"Embed + retrieve + LLM inference",
                     p4['avg_total_rag_sec']))

    col_w = [14, 26, 42, 12]
    header = (f"{'Phase':<{col_w[0]}}{'Description':<{col_w[1]}}"
              f"{'Details':<{col_w[2]}}{'Latency':>{col_w[3]}}")
    print(f"\n  {header}")
    print(f"  {'─' * sum(col_w)}")

    for phase, desc, details, secs in rows:
        print(f"  {phase:<{col_w[0]}}{desc:<{col_w[1]}}"
              f"{details:<{col_w[2]}}{ms(secs):>{col_w[3]}}")

    # Sub-breakdown for Phase 3
    print(f"\n  Phase 3 Sub-breakdown:")
    print(f"    {'Embed (sentence-transformer)':<40} "
          f"{ms(p3['cache_miss']['avg_embed_sec'])}")
    print(f"    {'DB Search (pgvector cosine)':<40} "
          f"{ms(p3['cache_miss']['avg_search_sec'])}")
    print(f"    {'Genre filter speedup':<40} "
          f"{p3['full_scan_sec'] / p3['genre_filtered_sec']:.1f}×")
    print(f"    {'Cache hit rate (session)':<40} "
          f"{p3['session_hit_rate_pct']:.0f}%")

    if not p4.get('phase4_skipped'):
        print(f"\n  Phase 4 Sub-breakdown:")
        print(f"    {'Query embed':<40} {ms(p4['avg_embed_sec'])}")
        print(f"    {'pgvector retrieve':<40} {ms(p4['avg_retrieve_sec'])}")
        print(f"    {'LLM inference (Claude Haiku)':<40} {ms(p4['avg_llm_sec'])}")

    # Per-query online latency (what users actually feel)
    online_miss = (p3['cache_miss']['avg_total_sec'] +
                   (p4.get('avg_llm_sec', 0) if not p4.get('phase4_skipped') else 0))
    online_hit  = (p3['cache_hit']['avg_total_sec'] +
                   (p4.get('avg_llm_sec', 0) if not p4.get('phase4_skipped') else 0))

    print(f"\n  {'─' * sum(col_w)}")
    print(f"  {'User-facing latency (cold)':<{col_w[0]+col_w[1]+col_w[2]}} "
          f"{ms(online_miss):>{col_w[3]}}")
    print(f"  {'User-facing latency (cached)':<{col_w[0]+col_w[1]+col_w[2]}} "
          f"{ms(online_hit):>{col_w[3]}}")

    separator()

    # Save JSON report
    report = {
        'phase1': p1,
        'phase2': {k: v for k, v in p2.items() if k != '__model__'},
        'phase3': p3,
        'phase4': p4,
        'summary': {
            'user_latency_cold_ms':   round(online_miss * 1000, 1),
            'user_latency_cached_ms': round(online_hit  * 1000, 1),
        }
    }
    report_path = "overall_latency_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📄 Full report saved → {report_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    separator("VibeLens — Pipeline Performance Benchmark")
    print(f"  Queries : {len(BENCHMARK_QUERIES)}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  DB host : {DB_CONFIG['host']}")

    p1 = phase1_data_ingestion()
    p2_results, model = phase2_feature_engineering()
    p3 = phase3_vector_search(model)
    p4 = phase4_rag_generation(model)

    print_summary(p1, p2_results, p3, p4)
