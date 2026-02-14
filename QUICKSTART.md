# VibeLens Quick Start Guide

## 🎯 10-Minute Setup Guide

### Prerequisites

✅ **Required**:
- Python 3.8+
- 16GB RAM (recommended)
- Aiven PostgreSQL account
- MovieLens 25M dataset
- TMDB crawler data

⚠️ **Optional**:
- AWS account (if using S3)
- GPU (speeds up embedding generation)

---

## 📥 Step 1: Get Project Code

### Option A: Clone from GitHub
```bash
git clone https://github.com/YourUsername/VibeCurators.git
cd VibeCurators
```

### Option B: Use provided files
You should have received these files:
- `etl_spark.py` - Spark ETL script
- `generate_embeddings.py` - Vectorization script
- `load_to_postgres.py` - Database import
- `search_movies.py` - Search engine
- `requirements.txt` - Dependencies list
- `.env.example` - Config template
- `run_pipeline.sh` - One-click execution script
- `README.md` - Full documentation

---

## 🔧 Step 2: Environment Setup (5 minutes)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
nano .env  # Or use any other editor
```

**Edit `.env` file**:
```bash
# Fill in your Aiven PostgreSQL information
PG_HOST=your-project.aivencloud.com
PG_PORT=12345
PG_DATABASE=defaultdb
PG_USER=avnadmin
PG_PASSWORD=xxxxx

# Keep defaults for everything else
```

---

## 📦 Step 3: Prepare Data (5 minutes)

### 3.1 Create Directory Structure
```bash
mkdir -p data/raw/movielens data/raw/tmdb data/preprocessed
```

### 3.2 Download MovieLens 25M
```bash
# Option A: Direct download
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
mv ml-25m/* data/raw/movielens/
rm -rf ml-25m ml-25m.zip

# Option B: If already downloaded
# Extract to data/raw/movielens/ directory
```

**Verify files**:
```bash
ls -lh data/raw/movielens/
# You should see:
# - ratings.csv (~900MB)
# - movies.csv
# - links.csv (~2MB)
```

### 3.3 Place TMDB Data
```bash
# Place your TMDB JSON files in this directory
cp /path/to/your/tmdb/*.json data/raw/tmdb/

# Or if single file
cp /path/to/your/11.json data/raw/tmdb/
```

---

## 🚀 Step 4: Run Pipeline (10-15 minutes)

### Option A: One-Click Run (Recommended)
```bash
./run_pipeline.sh
```

This automatically runs:
1. Spark ETL (3-5 minutes)
2. Embedding generation (5-10 minutes)
3. PostgreSQL import (2-5 minutes)

### Option B: Manual Step-by-Step
```bash
# Step 1: ETL
python etl_spark.py
# Output: data/preprocessed/movie_soup.parquet

# Step 2: Embedding
python generate_embeddings.py
# Output: data/preprocessed/movies_with_embeddings.parquet

# Step 3: PostgreSQL
python load_to_postgres.py
# Output: 'movies' table in Aiven database
```

---

## 🔍 Step 5: Start Searching!

```bash
python search_movies.py
```

**Example conversation**:
```
=== VibeLens Interactive Search ===
Enter your movie vibe description (or 'quit' to exit)

Search > dark sci-fi about loneliness and existential crisis
Year filter (e.g., '>= 2000', or press Enter to skip): >= 2010

======================================================================
Search Query: 'dark sci-fi about loneliness and existential crisis'
======================================================================

1. Blade Runner 2049 (2017)
   Genres: Science Fiction, Thriller
   Ratings: 5,432 reviews, avg 4.2/5.0
   Similarity: 0.872

2. Arrival (2016)
   Genres: Science Fiction, Drama
   Ratings: 4,321 reviews, avg 4.3/5.0
   Similarity: 0.843

3. Her (2013)
   Genres: Science Fiction, Romance, Drama
   Ratings: 3,876 reviews, avg 4.1/5.0
   Similarity: 0.821

Search > 
```

---

## 🎉 Success Indicators

If you see the search results above, the system is working!

### Verification Checklist
- ✅ ETL generated `movie_soup.parquet`
- ✅ Embedding generated `movies_with_embeddings.parquet`
- ✅ PostgreSQL has `movies` table
- ✅ Can perform semantic search

---

## 🛠️ Quick Fixes for Common Issues

### Issue 1: Spark Out of Memory
```python
# Edit etl_spark.py, line 33
.config("spark.driver.memory", "4g")  # Change to 4g or 6g
```

### Issue 2: PostgreSQL Connection Timeout
```bash
# Check Aiven console
# Services → your-postgres → Overview → Allowed IP addresses
# Add your public IP (find it at https://whatismyip.com)
```

### Issue 3: TMDB Data Format Error
```python
# Edit etl_spark.py, line 99
# Check if TMDB schema matches your data
# Adjust StructType definition if fields differ
```

### Issue 4: Embedding Generation Too Slow
```bash
# Option A: Reduce batch size
# Edit generate_embeddings.py, last line
BATCH_SIZE = 16  # Change to 16

# Option B: Use smaller dataset for testing
# Edit etl_spark.py, line 256
'min_ratings_threshold': 500  # Change to 500, reduces movie count
```

---

## 📊 Expected Performance

### Processing Time (16GB RAM, M1 chip)
- **ETL**: 3-5 minutes
- **Embedding**: 5-10 minutes (~15,000 movies)
- **PostgreSQL**: 2-5 minutes
- **Total**: 10-20 minutes

### Search Performance
- **First query**: ~200ms (model loading)
- **Subsequent queries**: ~50-100ms

### Data Scale
- **Raw data**: ~1.5GB
- **Processed**: ~500MB (Parquet + PostgreSQL)
- **Movies**: ~15,000 (ratings >= 100)

---

## 🎓 Course Presentation Tips

### Demo Flow
1. **Introduce problem**: Traditional keyword search vs semantic search
2. **Show architecture**: Spark ETL → Embeddings → pgvector
3. **Live demo**: 
   - Query "dark sci-fi about loneliness"
   - Compare PostgreSQL performance (with/without HNSW index)
4. **Performance metrics**: 
   - ETL processing time for 900MB data
   - HNSW index speedup (10-100x)
5. **Big data techniques**: 
   - Spark distributed processing
   - Bulk COPY import
   - Vector index optimization

### Visualization Suggestions
```python
# Generate performance comparison chart
import matplotlib.pyplot as plt

# Without index vs with index
times = [2.5, 0.08]  # seconds
labels = ['Linear Scan', 'HNSW Index']
plt.bar(labels, times)
plt.ylabel('Query Time (seconds)')
plt.title('Search Performance Comparison')
plt.show()
```

---

## 📞 Need Help?

### Debug Mode
```bash
# Enable verbose logging
export SPARK_LOG_LEVEL=INFO
python etl_spark.py

# Check data quality
python -c "
import pandas as pd
df = pd.read_parquet('data/preprocessed/movie_soup.parquet')
print(df.info())
print(df.head())
"
```

### Check Database
```bash
# Connect to PostgreSQL
export DATABASE_URL="postgresql://user:pass@host:port/db"
psql $DATABASE_URL

# In psql
SELECT COUNT(*) FROM movies;
SELECT title, year FROM movies LIMIT 5;
\d movies  -- View table structure
```

---

## ✅ Next Steps

After completing basic functionality, consider:

1. **Add LLM layer** (Option C)
   - Use OpenAI API to generate recommendation reasons
   - Implement multi-hop reasoning

2. **Web interface**
   - Build API with Flask/FastAPI
   - Create frontend with React

3. **Performance tuning**
   - Experiment with different HNSW parameters
   - Test different embedding models

---

**Good luck with your project!** 🚀

For more details, see `README.md` and `PROJECT_STRUCTURE.md`.
