#!/usr/bin/env python3
"""
VibeLens — Step 1: Export DB → Local Parquet
"""

import os
import time
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host':     os.getenv('PG_HOST'),
    'port':     os.getenv('PG_PORT', '5432'),
    'database': os.getenv('PG_DATABASE', 'vibelens'),
    'user':     os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD'),
}

OUTPUT_PATH = './data/preprocessed/movies_with_embeddings.parquet'
os.makedirs('./data/preprocessed', exist_ok=True)

print("=" * 55)
print("  VibeLens — DB Export (READ-ONLY, 不影响云上数据)")
print("=" * 55)
print(f"  Host : {DB_CONFIG['host']}")
print(f"  DB   : {DB_CONFIG['database']}\n")

conn = psycopg2.connect(**DB_CONFIG)
cur  = conn.cursor()

# total rows
cur.execute('SELECT COUNT(*) FROM movies')
total = cur.fetchone()[0]
print(f"  Find {total:,} movies, starting export...\n")

start = time.time()

cur.execute("""
    SELECT "movieId", "title", "year", "genres", "tmdb_genres",
           "num_ratings", "avg_rating", "tmdb_rating", "soup", "embedding"
    FROM movies
""")
rows = cur.fetchall()
cur.close()
conn.close()

fetch_time = time.time() - start
print(f"  ✓ Fetch completed  ({fetch_time:.1f}s)")

# transform DataFrame
cols = ['movieId','title','year','genres','tmdb_genres',
        'num_ratings','avg_rating','tmdb_rating','soup','embedding']
df = pd.DataFrame(rows, columns=cols)

# embedding: list[float] → list (keep format, consistent with generate_embeddings.py)
df['embedding'] = df['embedding'].apply(lambda x: list(x) if x is not None else None)

save_start = time.time()
df.to_parquet(OUTPUT_PATH, index=False)
save_time = time.time() - save_start

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
total_time = time.time() - start

print(f"  ✓ Save completed  ({save_time:.1f}s)")
print(f"\n  Output file : {OUTPUT_PATH}")
print(f"  File size : {size_mb:.1f} MB")
print(f"  Total time : {total_time:.1f}s")
print(f"\n  ✅ Completed! Now you can run benchmark_pipeline.py")