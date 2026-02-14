#!/bin/bash

# VibeLens Complete Pipeline Execution Script
# Automatically run ETL → Embedding → PostgreSQL import

set -e  # Exit immediately on error

echo "==================================================="
echo "    VibeLens - Complete Pipeline Execution"
echo "==================================================="

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check .env file
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Create data directories
echo ""
echo "Step 0: Creating data directories..."
mkdir -p data/raw/movielens data/raw/tmdb data/preprocessed
echo "✓ Directories created"

# Check input files
echo ""
echo "Step 0.5: Checking input files..."
if [ ! -f "data/raw/movielens/ratings.csv" ]; then
    echo "Warning: ratings.csv not found in data/raw/movielens/"
    echo "Please download MovieLens 25M dataset"
    exit 1
fi
if [ ! -f "data/raw/tmdb/11.json" ] && [ -z "$(ls -A data/raw/tmdb/*.json 2>/dev/null)" ]; then
    echo "Warning: TMDB JSON files not found in data/raw/tmdb/"
    echo "Please add your TMDB data"
    exit 1
fi
echo "✓ Input files verified"

# Step 1: Spark ETL
echo ""
echo "==================================================="
echo "Step 1: Running Spark ETL..."
echo "==================================================="
python etl_spark.py

if [ ! -f "data/preprocessed/movie_soup.parquet" ]; then
    echo "Error: ETL failed - movie_soup.parquet not created"
    exit 1
fi
echo "✓ ETL complete"

# Step 2: Generate Embeddings
echo ""
echo "==================================================="
echo "Step 2: Generating Embeddings..."
echo "==================================================="
python generate_embeddings.py

if [ ! -f "data/preprocessed/movies_with_embeddings.parquet" ]; then
    echo "Error: Embedding generation failed"
    exit 1
fi
echo "✓ Embeddings generated"

# Step 3: Load to PostgreSQL
echo ""
echo "==================================================="
echo "Step 3: Loading to PostgreSQL..."
echo "==================================================="
python load_to_postgres.py

echo "✓ PostgreSQL loading complete"

# Complete
echo ""
echo "==================================================="
echo "    ✅ Pipeline Complete!"
echo "==================================================="
echo ""
echo "You can now search movies with:"
echo "  python search_movies.py"
echo ""
echo "Or test the database with:"
echo "  psql \$DATABASE_URL -c 'SELECT COUNT(*) FROM movies;'"
echo ""
