#!/bin/bash

# VibeLens Environment Setup Script
echo "=== VibeLens Environment Setup ==="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Core dependencies
echo "Installing core dependencies..."
pip install pandas==2.1.0
pip install numpy==1.24.3
pip install psycopg2-binary==2.9.7
pip install pgvector==0.2.3

# Spark (using PySpark)
echo "Installing PySpark..."
pip install pyspark==3.5.0

# Sentence Transformers
echo "Installing Sentence Transformers..."
pip install sentence-transformers==2.2.2

# AWS SDK
echo "Installing AWS SDK..."
pip install boto3==1.28.0
pip install awscli==1.29.0

# Data processing
echo "Installing data processing tools..."
pip install pyarrow==13.0.0  # Parquet support
pip install tqdm==4.66.0      # Progress bar

# Optional: Jupyter (for debugging)
echo "Installing Jupyter (optional)..."
pip install jupyter==1.0.0

echo "=== Installation Complete ==="
echo "Activate environment with: source venv/bin/activate"
