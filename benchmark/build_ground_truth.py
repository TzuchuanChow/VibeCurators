#!/usr/bin/env python3
"""
VibeLens Ground Truth Builder
==============================
Generates evaluation ground truth using MovieLens genome-tags.

Pipeline:
  1. Load genome-scores.csv + genome-tags.csv + movies.csv
  2. Filter to the same movie subset used in VibeLens
     (movies with >= 100 ratings that survived the ETL TMDB join)
  3. For each vibe query, use sentence-transformers cosine similarity
     to find the most relevant genome-tags  (no external API needed)
  4. Look up all movies where genome-score > 0.8 for those tags
  5. Save as test_queries_genome.json  ← use this in evaluate_vibelens.py

Usage:
  python build_ground_truth.py --movielens-dir ../data/raw/movielens
  python build_ground_truth.py --movielens-dir ../data/raw/movielens --score-threshold 0.8
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── Vibe queries to evaluate ──────────────────────────────────────────────────
VIBE_QUERIES = [
    {"query": "dark sci-fi about time loops",                          "genre_hint": "Science Fiction"},
    {"query": "isolated researcher finds something mysterious in ice",  "genre_hint": "Science Fiction"},
    {"query": "retro-futuristic heist movie",                          "genre_hint": None},
    {"query": "animated family movie about growing up",                 "genre_hint": "Animation"},
    {"query": "scary ghost haunted house horror",                       "genre_hint": "Horror"},
    {"query": "romantic comedy about mistaken identity",                "genre_hint": "Comedy"},
    {"query": "psychological thriller with mind-bending twists",        "genre_hint": "Thriller"},
    {"query": "space exploration with beautiful visuals",               "genre_hint": "Science Fiction"},
    {"query": "war movie about soldiers surviving against all odds",    "genre_hint": "Action"},
    {"query": "coming of age teenage drama",                            "genre_hint": "Drama"},
    {"query": "survival in the wilderness",                             "genre_hint": None},
    {"query": "time travel romance",                                    "genre_hint": "Romance"},
    {"query": "funny road trip with unlikely friends",                  "genre_hint": "Comedy"},
    {"query": "emotional drama about family grief and loss",            "genre_hint": "Drama"},
    {"query": "cyberpunk dystopian future city",                        "genre_hint": "Science Fiction"},
    {"query": "zombie apocalypse survival horror",                      "genre_hint": "Horror"},
    {"query": "underdog sports team wins championship",                 "genre_hint": None},
    {"query": "true crime detective murder mystery",                    "genre_hint": "Thriller"},
    {"query": "anime fantasy adventure with magic and friendship",      "genre_hint": "Animation"},
    {"query": "superhero learning to use powers for first time",        "genre_hint": "Action"},
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load MovieLens genome data
# ─────────────────────────────────────────────────────────────────────────────

def load_genome_data(movielens_dir: str):
    """
    Load and join genome-scores + genome-tags + movies.

    Returns:
        genome_df : movieId | tagId | tag | score | title
        all_tags  : sorted list of all unique tag strings
    """
    print("\n=== Loading MovieLens Genome Data ===")

    scores_df = pd.read_csv(os.path.join(movielens_dir, "genome-scores.csv"))
    tags_df   = pd.read_csv(os.path.join(movielens_dir, "genome-tags.csv"))
    movies_df = pd.read_csv(os.path.join(movielens_dir, "movies.csv"))

    scores_df = scores_df.rename(columns={"relevance": "score"})

    genome_df = (
        scores_df
        .merge(tags_df,                         on="tagId",   how="left")
        .merge(movies_df[["movieId", "title"]],  on="movieId", how="left")
    )

    all_tags = sorted(tags_df["tag"].str.lower().unique().tolist())

    print(f"  Movies with genome data : {genome_df['movieId'].nunique():,}")
    print(f"  Total genome tags       : {len(all_tags):,}")
    print(f"  Total score rows        : {len(genome_df):,}")

    return genome_df, all_tags


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Get the same movie subset as VibeLens (from PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────

def get_vibelens_movie_ids(db_config: dict) -> set:
    """
    Fetch all movieIds currently in the VibeLens PostgreSQL database.
    This is the exact subset that survived:
      - ratings >= 100 filter  (etl_spark.py)
      - TMDB join              (etl_spark.py)
    Ground truth is restricted to this subset because the system
    can only return movies that exist in the DB.
    """
    print("\n=== Fetching VibeLens Movie Subset from PostgreSQL ===")
    conn   = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute('SELECT "movieId" FROM movies')
    ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    print(f"  Movies in VibeLens DB : {len(ids):,}")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Map vibe query → genome tags via semantic similarity
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache: tag embeddings are expensive, only compute once per run
_tag_embeddings_cache = {}

def map_vibe_to_tags(
    query:      str,
    all_tags:   list,
    top_n_tags: int = 10,
    model=None,
) -> list:
    """
    Select the most relevant genome-tags for a vibe query using cosine
    similarity between sentence embeddings (all-MiniLM-L6-v2).

    No external API needed — reuses the same model already used for search.

    Steps:
      1. Encode the query into a 384-D embedding
      2. Encode all genome-tags once (cached for subsequent queries)
      3. Return the top_n_tags tags by cosine similarity

    Returns:
        List of tag strings (lowercase, as they appear in genome-tags.csv)
    """
    # Encode and L2-normalise the query
    query_emb = model.encode([query], convert_to_numpy=True)[0]
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)

    # Encode all tags once, cache by list identity (stable within one run)
    cache_key = id(all_tags)
    if cache_key not in _tag_embeddings_cache:
        print("  (encoding all genome tags — one-time cost ~5s)...")
        tag_embs = model.encode(
            all_tags,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=False,
        )
        norms = np.linalg.norm(tag_embs, axis=1, keepdims=True) + 1e-9
        _tag_embeddings_cache[cache_key] = tag_embs / norms

    tag_embs_norm = _tag_embeddings_cache[cache_key]  # (N_tags, 384)

    # Cosine similarity = dot product of L2-normalised vectors
    scores  = tag_embs_norm @ query_emb               # (N_tags,)
    top_idx = np.argsort(scores)[::-1][:top_n_tags]

    return [all_tags[i] for i in top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Look up relevant movies by genome score
# ─────────────────────────────────────────────────────────────────────────────

def get_relevant_movies(
    genome_df:       pd.DataFrame,
    tags:            list,
    vibelens_ids:    set,
    score_threshold: float = 0.8,
) -> list:
    """
    Find all movies where genome-score > threshold for ANY of the given tags,
    restricted to the VibeLens movie subset.

    A movie is "relevant" if it scores > threshold on at least one matched tag.
    This is the ground truth set — the denominator for Recall@K.

    Returns:
        List of movie title strings, sorted by max score descending.
    """
    if not tags:
        return []

    mask = (
        genome_df["tag"].str.lower().isin([t.lower() for t in tags])
        & (genome_df["score"] > score_threshold)
        & (genome_df["movieId"].isin(vibelens_ids))
    )

    matched = genome_df[mask]

    # One row per movie: keep the highest score across all matched tags
    agg = (
        matched
        .groupby(["movieId", "title"])["score"]
        .max()
        .reset_index()
        .sort_values("score", ascending=False)
    )

    return agg["title"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build full test set and save
# ─────────────────────────────────────────────────────────────────────────────

def build_ground_truth(
    movielens_dir:   str,
    db_config:       dict,
    score_threshold: float = 0.8,
    top_n_tags:      int   = 10,
    save_path:       str   = "./test_queries_genome.json",
) -> list:

    # 1. Load genome data
    genome_df, all_tags = load_genome_data(movielens_dir)

    # 2. Get VibeLens movie subset
    vibelens_ids = get_vibelens_movie_ids(db_config)

    # 3. Load embedding model once — reused for all tag mappings
    print("\n=== Loading Embedding Model ===")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded.")

    print(f"\n=== Mapping {len(VIBE_QUERIES)} Vibe Queries → Genome Tags ===")
    print(f"    Score threshold : {score_threshold}")
    print(f"    Tags per query  : {top_n_tags}\n")

    test_queries = []

    for i, item in enumerate(VIBE_QUERIES, 1):
        query      = item["query"]
        genre_hint = item.get("genre_hint")

        print(f"[{i:02d}/{len(VIBE_QUERIES)}] {query}")

        # 3a. Vibe → genome tags (sentence-transformer cosine similarity)
        matched_tags = map_vibe_to_tags(query, all_tags, top_n_tags, model=model)
        print(f"         Tags ({len(matched_tags)}): {matched_tags[:5]}"
              f"{'...' if len(matched_tags) > 5 else ''}")

        # 3b. Tags → relevant movies (genome score filter + VibeLens subset)
        relevant_titles = get_relevant_movies(
            genome_df, matched_tags, vibelens_ids, score_threshold
        )
        print(f"         Relevant movies (score > {score_threshold}): "
              f"{len(relevant_titles)}")
        if relevant_titles:
            print(f"         Sample : {relevant_titles[:4]}")

        test_queries.append({
            "query":           query,
            "genre_hint":      genre_hint,
            "matched_tags":    matched_tags,
            "score_threshold": score_threshold,
            "relevant_titles": relevant_titles,
            "num_relevant":    len(relevant_titles),
        })

    # 4. Save JSON (create output directory if needed)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(test_queries, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(test_queries)} queries → {save_path}")

    # 5. Summary stats
    sizes = [q["num_relevant"] for q in test_queries]
    print(f"\n=== Ground Truth Summary ===")
    print(f"  Queries with 0 relevant movies  : {sum(1 for s in sizes if s == 0)}")
    print(f"  Avg relevant movies per query   : {sum(sizes)/len(sizes):.1f}")
    print(f"  Min / Max relevant              : {min(sizes)} / {max(sizes)}")
    print(f"  Total unique relevant movies    : "
          f"{len(set(t for q in test_queries for t in q['relevant_titles'])):,}")

    return test_queries


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VibeLens Ground Truth Builder")
    parser.add_argument("--movielens-dir",   default="../data/raw/movielens",
                        help="Directory containing genome-scores.csv, genome-tags.csv, movies.csv")
    parser.add_argument("--score-threshold", type=float, default=0.8,
                        help="Min genome-score to count as relevant (default: 0.8)")
    parser.add_argument("--top-n-tags",      type=int,   default=10,
                        help="Genome-tags selected per query (default: 10)")
    parser.add_argument("--output",          default="./test_queries_genome.json",
                        help="Output path for ground truth JSON")
    args = parser.parse_args()

    db_config = {
        "host":     os.getenv("PG_HOST"),
        "port":     os.getenv("PG_PORT", "5432"),
        "database": os.getenv("PG_DATABASE", "vibelens"),
        "user":     os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD"),
    }

    build_ground_truth(
        movielens_dir   = args.movielens_dir,
        db_config       = db_config,
        score_threshold = args.score_threshold,
        top_n_tags      = args.top_n_tags,
        save_path       = args.output,
    )

    print(f"\nNext step — run evaluation with this ground truth:")
    print(f"  python evaluate_vibelens.py --queries-file {args.output}")


if __name__ == "__main__":
    main()
