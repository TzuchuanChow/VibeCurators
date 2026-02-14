# VibeLens: Semantic Movie Recommender

A vector embedding-based movie recommendation system using RAG and PostgreSQL+pgvector for UC Riverside's Big Data Management course.

## 📋 Project Overview

VibeLens enables users to find movies through **natural language descriptions** (e.g., "dark sci-fi about loneliness") instead of traditional keyword matching, leveraging vector embeddings and semantic search.

### Core Tech Stack

- **Data Processing**: PySpark (distributed ETL)
- **Vectorization**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Database**: PostgreSQL + pgvector (Aiven hosted)
- **Indexing**: HNSW (Hierarchical Navigable Small World)
- **Data Sources**: MovieLens 25M + TMDB API

---

## 🏗️ System Architecture

```
[Offline Preparation - One-time Setup]
MovieLens (25M) + TMDB (157K movies)
    ↓
PySpark ETL (merge, filter, generate Movie Soup)
    ↓
Sentence-Transformers (generate 384-D embeddings)
    ↓
PostgreSQL + pgvector (HNSW index)

[Online Queries - Real-time]
User: "dark sci-fi about loneliness"
    ↓
Sentence-Transformers (query → vector)
    ↓
pgvector (HNSW fast retrieval)
    ↓
Return: Top-5 movies + similarity scores
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/YourUsername/VibeCurators.git
cd VibeCurators

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file and add your Aiven PostgreSQL credentials
```

### 2. Data Preparation

```bash
# Create data directories
mkdir -p data/raw/movielens data/raw/tmdb data/preprocessed

# Download MovieLens 25M (~250MB)
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip -d data/raw/movielens/

# Place your TMDB crawler data in data/raw/tmdb/
# (Ensure your JSON files are in this directory)
```

### 3. Run ETL Pipeline

```bash
# Step 1: Spark ETL (merge data + generate Movie Soup)
python etl_spark.py
# Output: data/preprocessed/movie_soup.parquet

# Step 2: Generate Embeddings (~5-10 minutes)
python generate_embeddings.py
# Output: data/preprocessed/movies_with_embeddings.parquet

# Step 3: Import to PostgreSQL (~2-5 minutes)
python load_to_postgres.py
# Creates 'movies' table and HNSW index in Aiven PostgreSQL
```

### 4. Search Movies

```bash
# Start interactive search
python search_movies.py

# Example query
Search > dark sci-fi about loneliness
Year filter (e.g., '>= 2000', or press Enter to skip): >= 2010

# Output:
# 1. Blade Runner 2049 (2017)
#    Genres: Sci-Fi, Thriller
#    Ratings: 3,456 reviews, avg 4.2/5.0
#    Similarity: 0.872
# ...
```

---

## 📊 Data Statistics

### Raw Data
- **MovieLens 25M**: 25,000,095 ratings, 62,423 movies
- **TMDB**: 157,180 movies, 12,820 with reviews

### Processed Data
- **Filter threshold**: Ratings >= 100
- **Final dataset**: ~15,000 movies
- **Embedding dimension**: 384-D
- **Database size**: ~500MB

---

## 🔧 Configuration

### ETL Parameters (`etl_spark.py`)

```python
config = {
    'min_ratings_threshold': 100,  # Filter movies with < 100 ratings
    'movielens_movies_path': './data/raw/movielens/movies.csv',
    'movielens_ratings_path': './data/raw/movielens/ratings.csv',
    'movielens_links_path': './data/raw/movielens/links.csv',
    'tmdb_path': './data/raw/tmdb/*.json',
    'output_parquet_path': './data/preprocessed/movie_soup.parquet'
}
```

### Embedding Parameters (`generate_embeddings.py`)

```python
MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-D, fast
# Or use 'all-mpnet-base-v2'     # 768-D, more accurate but slower

BATCH_SIZE = 32  # Adjust based on memory (16-64)
```

### HNSW Index Parameters (`load_to_postgres.py`)

```python
m = 16                # HNSW max connections per layer
ef_construction = 64  # Search range during index construction

# Larger values → More accurate but slower
# Recommended range: m=8-64, ef_construction=32-200
```

---

## 📈 Performance Metrics

### ETL Performance (16GB RAM, Apple M1)
- **Spark ETL**: ~3 minutes
- **Embedding generation**: ~8 minutes (15,000 movies @ 30 texts/sec)
- **PostgreSQL import**: ~2 minutes
- **HNSW index creation**: ~1 minute

### Query Performance
- **Without index**: ~2-5 seconds (linear scan)
- **With HNSW index**: ~50-200ms (logarithmic complexity)
- **Speedup**: ~10-100x

---

## 🎓 Big Data Management Course Highlights

This project demonstrates the following "Big Data Management" concepts:

1. **Distributed Data Processing**: PySpark ETL
2. **Index Optimization**: HNSW vs linear scan (O(log N) vs O(N))
3. **Bulk Data Loading**: COPY vs INSERT
4. **Hybrid Queries**: SQL filtering + vector similarity
5. **Data Partitioning**: Parquet columnar storage
6. **Hot/Cold Data Separation**: Vectors in DB, full reviews in filesystem

---

## 🛠️ Troubleshooting

### Spark Out of Memory
```python
# Adjust memory in etl_spark.py
.config("spark.driver.memory", "4g")  # Reduce to 4g
```

### PostgreSQL Connection Failed
```bash
# Check Aiven whitelist
# Aiven Console → Services → your-postgres → Overview → Allowed IP addresses
# Add your public IP
```

### Embedding Generation Too Slow
```python
# Reduce batch size
BATCH_SIZE = 16  # Change from 32 to 16
```

---

## 📚 References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Sentence-Transformers Models](https://www.sbert.net/docs/pretrained_models.html)
- [PySpark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)

---

## 👥 Team Members

- [Your Name] - ETL Pipeline, Vector Search
- [Team Member 2]
- [Team Member 3]

---

## 📄 License

MIT License
