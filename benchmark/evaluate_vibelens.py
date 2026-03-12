#!/usr/bin/env python3
"""
VibeLens Evaluation — Recall@5
================================
Evaluates Semantic Search (pgvector HNSW) vs TF-IDF (PostgreSQL FTS).

Ground truth : test_queries_genome.json  (built by build_ground_truth.py)
               relevant_titles = all movies in VibeLens DB with
               genome-score > 0.8 on query-matched tags

True Positive : edit distance between normalized titles <= threshold (default 3)
                tolerates punctuation / subtitle differences, rejects false matches

Recall@K :  |TP in top-K|  /  |relevant_titles|
            numerator   = how many ground-truth movies the system found
            denominator = full genome ground-truth set (not capped at K)

Usage:
  python evaluate_vibelens.py
  python evaluate_vibelens.py --queries-file ./test_queries_genome.json --k 5
  python evaluate_vibelens.py --edit-threshold 2
"""

import time
import json
import argparse
import os
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Edit-distance TP matching
# ─────────────────────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """
    Standard dynamic-programming Levenshtein distance.
    No external library needed.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalize_title(title: str) -> str:
    """
    Normalize title for edit-distance comparison:
      1. Strip trailing year "(1995)" — MovieLens DB titles include it,
         genome movies.csv titles also include it, but they must both
         be stripped so short titles don't accumulate year-char penalties.
      2. Lowercase + remove punctuation/separators.

    Examples:
      "Toy Story (1995)"          → "toy story"
      "Spider-Man: Homecoming"    → "spider man homecoming"
      "Blade Runner 2049 (2017)"  → "blade runner 2049"
      "Se7en (1995)"              → "se7en"
    """
    import re
    t = title.strip()
    t = re.sub(r"\(\d{4}\)\s*$", "", t).strip()  # remove trailing (YYYY)
    t = t.lower()
    t = re.sub(r"[:\-–—]", " ", t)               # separators → space
    t = re.sub(r"[^\w\s]", "", t)                 # remove remaining punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_relevant(result_title: str, relevant_titles: list, threshold: int = 3) -> bool:
    """
    True Positive definition:
      edit_distance(normalize(result), normalize(ground_truth)) <= threshold

    threshold=3 rationale:
      - "Alien" vs "Aliens"           → dist=1  ✅ TP
      - "Se7en" vs "Seven"            → dist=2  ✅ TP
      - "Spider Man" vs "Spider-Man"  → dist=0  ✅ TP  (after normalize)
      - "Alien" vs "Alien Resurrection" → dist=13 ❌ not TP
      - "The Thing" vs "The Ring"     → dist=2  ⚠️  borderline — raise threshold to 4 if needed
    """
    nr = normalize_title(result_title)
    for gt in relevant_titles:
        ng = normalize_title(gt)
        if _edit_distance(nr, ng) <= threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Recall@K
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list, relevant: list, k: int, threshold: int = 3) -> float:
    """
    Recall@K = |TP in top-K| / |relevant|

    Numerator   : count of retrieved[:k] that are TP (edit-distance match)
    Denominator : full ground-truth set length (from genome, not capped at K)

    Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for t in retrieved[:k] if is_relevant(t, relevant, threshold))
    return hits / len(relevant)


# ─────────────────────────────────────────────────────────────────────────────
# Search engines
# ─────────────────────────────────────────────────────────────────────────────

# Import the production MovieSearchEngine built by the team.
# benchmark/ is one level below the project root, so we add .. to sys.path.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from search_movies import MovieSearchEngine


class SemanticSearchEngine:
    """
    Thin wrapper around the production MovieSearchEngine so the evaluation
    loop uses the same embedding model, cache, and HNSW index as the real app.
    """

    def __init__(self, db_config):
        self._engine = MovieSearchEngine(db_config)

    def search(self, query_text: str, top_k: int, genre: str = None) -> tuple:
        t0 = time.time()

        # get_query_embedding handles cache hit/miss internally
        query_embedding = self._engine.get_query_embedding(query_text)
        t_embed = time.time() - t0

        emb_list = query_embedding.tolist()
        t1 = time.time()

        if genre:
            self._engine.cursor.execute("""
                SELECT title FROM movies
                WHERE  tmdb_genres LIKE %s
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
            """, (f"%{genre}%", emb_list, top_k))
        else:
            self._engine.cursor.execute("""
                SELECT title FROM movies
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
            """, (emb_list, top_k))

        titles = [r[0] for r in self._engine.cursor.fetchall()]

        return titles, {
            "embed_ms":   round(t_embed * 1000, 2),
            "search_ms":  round((time.time() - t1) * 1000, 2),
            "total_ms":   round((time.time() - t0) * 1000, 2),
            "from_cache": self._engine.cache_stats["hits"] > 0,
        }

    @property
    def cache_hit_rate(self):
        s = self._engine.cache_stats
        total = s["hits"] + s["misses"]
        return s["hits"] / total * 100 if total else 0.0

    def close(self):
        self._engine.close()


class TFIDFSearchEngine:
    """PostgreSQL full-text search baseline (mirrors Spark ML TF-IDF notebook)."""

    def __init__(self, db_config):
        self.conn   = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        self._ensure_index()

    def _ensure_index(self):
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS movies_soup_fts_idx
            ON movies USING gin(to_tsvector('english', soup));
        """)
        self.conn.commit()

    def search(self, query_text: str, top_k: int, genre: str = None) -> tuple:
        t0 = time.time()
        words    = [w for w in query_text.split() if len(w) > 2]
        ts_query = " | ".join(words) if words else "movie"

        try:
            if genre:
                self.cursor.execute("""
                    SELECT title,
                           ts_rank(to_tsvector('english', soup),
                                   to_tsquery('english', %s)) AS rank
                    FROM   movies
                    WHERE  tmdb_genres LIKE %s
                      AND  to_tsvector('english', soup) @@ to_tsquery('english', %s)
                    ORDER  BY rank DESC
                    LIMIT  %s
                """, (ts_query, f"%{genre}%", ts_query, top_k))
            else:
                self.cursor.execute("""
                    SELECT title,
                           ts_rank(to_tsvector('english', soup),
                                   to_tsquery('english', %s)) AS rank
                    FROM   movies
                    WHERE  to_tsvector('english', soup) @@ to_tsquery('english', %s)
                    ORDER  BY rank DESC
                    LIMIT  %s
                """, (ts_query, ts_query, top_k))
            titles = [r[0] for r in self.cursor.fetchall()]
        except Exception:
            self.conn.rollback()
            titles = []

        return titles, {"total_ms": round((time.time() - t0) * 1000, 2)}

    def close(self):
        self.cursor.close()
        self.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    db_config:      dict,
    test_queries:   list,
    k:              int = 5,
    edit_threshold: int = 3,
) -> dict:

    print(f"\n{'='*62}")
    print(f"  VibeLens Evaluation")
    print(f"  Queries        : {len(test_queries)}")
    print(f"  Recall@K       : K = {k}")
    print(f"  TP definition  : edit_distance(normalize(title)) <= {edit_threshold}")
    print(f"  Denominator    : full genome ground-truth set per query")
    print(f"{'='*62}\n")

    semantic = SemanticSearchEngine(db_config)
    tfidf    = TFIDFSearchEngine(db_config)

    per_query = []

    for i, item in enumerate(test_queries, 1):
        query    = item["query"]
        relevant = item["relevant_titles"]   # full genome ground truth
        genre    = item.get("genre_hint")
        n_rel    = len(relevant)

        print(f"[{i:02d}/{len(test_queries)}] {query[:52]:<52} "
              f"| {n_rel} relevant")

        # ── Semantic ──────────────────────────────────────────────────────
        sem_titles, sem_t = semantic.search(query, k, genre)
        sem_hits   = sum(1 for t in sem_titles if is_relevant(t, relevant, edit_threshold))
        sem_recall = round(sem_hits / n_rel, 4) if n_rel else 0.0

        # ── TF-IDF ────────────────────────────────────────────────────────
        tfidf_titles, tfidf_t = tfidf.search(query, k, genre)
        tfidf_hits   = sum(1 for t in tfidf_titles if is_relevant(t, relevant, edit_threshold))
        tfidf_recall = round(tfidf_hits / n_rel, 4) if n_rel else 0.0

        print(f"         Semantic  recall@{k}={sem_recall:.3f}  "
              f"({sem_hits}/{n_rel})  "
              f"{sem_t['total_ms']:6.1f}ms "
              f"({'cache' if sem_t['from_cache'] else 'encode'})")
        print(f"         TF-IDF    recall@{k}={tfidf_recall:.3f}  "
              f"({tfidf_hits}/{n_rel})  "
              f"{tfidf_t['total_ms']:6.1f}ms")

        # Debug: if no hits, show normalized titles to reveal mismatch format
        if sem_hits == 0 and n_rel > 0 and i <= 3:
            sem_norm = [normalize_title(t) for t in sem_titles[:2]]
            gt_norm  = [normalize_title(t) for t in relevant[:2]]
            print(f"         [debug] DB titles normalized  : {sem_norm}")
            print(f"         [debug] GT titles normalized  : {gt_norm}")

        per_query.append({
            "query":          query,
            "genre":          genre,
            "num_relevant":   n_rel,
            "matched_tags":   item.get("matched_tags", []),
            "semantic": {
                f"recall@{k}": sem_recall,
                "hits":        sem_hits,
                "total_ms":    sem_t["total_ms"],
                "embed_ms":    sem_t["embed_ms"],
                "search_ms":   sem_t["search_ms"],
                "from_cache":  sem_t["from_cache"],
                "top_results": sem_titles,
            },
            "tfidf": {
                f"recall@{k}": tfidf_recall,
                "hits":        tfidf_hits,
                "total_ms":    tfidf_t["total_ms"],
                "top_results": tfidf_titles,
            },
        })

    # ── Aggregate ─────────────────────────────────────────────────────────
    recall_key = f"recall@{k}"

    sem_recalls   = [r["semantic"][recall_key] for r in per_query]
    tfidf_recalls = [r["tfidf"][recall_key]    for r in per_query]
    sem_lats      = [r["semantic"]["total_ms"] for r in per_query]
    tfidf_lats    = [r["tfidf"]["total_ms"]    for r in per_query]

    avg_sem_recall   = round(sum(sem_recalls)   / len(sem_recalls),   4)
    avg_tfidf_recall = round(sum(tfidf_recalls) / len(tfidf_recalls), 4)
    avg_sem_lat      = round(sum(sem_lats)       / len(sem_lats),      2)
    avg_tfidf_lat    = round(sum(tfidf_lats)     / len(tfidf_lats),    2)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  RESULTS  |  {len(test_queries)} queries  |  K={k}  |  "
          f"edit_threshold={edit_threshold}")
    print(f"{'='*62}")
    print(f"  {'Metric':<28} {'Semantic':>12} {'TF-IDF':>12} {'Δ':>8}")
    print(f"  {'-'*60}")
    print(f"  {f'Avg Recall@{k}':<28} "
          f"{avg_sem_recall:>12.4f} "
          f"{avg_tfidf_recall:>12.4f} "
          f"{avg_sem_recall - avg_tfidf_recall:>+8.4f}")
    print(f"  {'Avg Latency (ms)':<28} "
          f"{avg_sem_lat:>12.1f} "
          f"{avg_tfidf_lat:>12.1f} "
          f"{avg_sem_lat - avg_tfidf_lat:>+8.1f}")
    print(f"  {'Cache Hit Rate':<28} "
          f"{semantic.cache_hit_rate:>11.1f}%")

    # ── Per-query breakdown table ──────────────────────────────────────────
    print(f"\n  Per-query Recall@{k}:")
    print(f"  {'Query':<50} {'Sem':>6} {'TFIDF':>6} {'#Rel':>5}")
    print(f"  {'-'*70}")
    for r in per_query:
        print(f"  {r['query'][:50]:<50} "
              f"{r['semantic'][recall_key]:>6.3f} "
              f"{r['tfidf'][recall_key]:>6.3f} "
              f"{r['num_relevant']:>5}")

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        "config": {
            "k":              k,
            "edit_threshold": edit_threshold,
            "num_queries":    len(test_queries),
        },
        "summary": {
            f"semantic_avg_recall@{k}":   avg_sem_recall,
            f"tfidf_avg_recall@{k}":      avg_tfidf_recall,
            "semantic_avg_latency_ms":    avg_sem_lat,
            "tfidf_avg_latency_ms":       avg_tfidf_lat,
            "semantic_cache_hit_rate":    round(semantic.cache_hit_rate, 1),
        },
        "per_query": per_query,
    }

    report_path = "./eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report → {report_path}")

    semantic.close()
    tfidf.close()
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VibeLens Recall@K Evaluation")
    parser.add_argument("--queries-file",   default="./test_queries_genome.json",
                        help="Ground truth JSON from build_ground_truth.py")
    parser.add_argument("--k",              type=int, default=5,
                        help="K for Recall@K (default: 5)")
    parser.add_argument("--edit-threshold", type=int, default=3,
                        help="Max edit distance to count as TP (default: 3)")
    args = parser.parse_args()

    db_config = {
        "host":     os.getenv("PG_HOST"),
        "port":     os.getenv("PG_PORT", "5432"),
        "database": os.getenv("PG_DATABASE", "vibelens"),
        "user":     os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD"),
    }

    with open(args.queries_file) as f:
        test_queries = json.load(f)
    print(f"  Loaded {len(test_queries)} queries from {args.queries_file}")

    run_evaluation(
        db_config      = db_config,
        test_queries   = test_queries,
        k              = args.k,
        edit_threshold = args.edit_threshold,
    )
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
