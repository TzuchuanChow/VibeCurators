"""
VibeLens Embedding Generation
Generate movie embeddings using Sentence-Transformers
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import os

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize Embedding Generator
        
        Args:
            model_name: Sentence-Transformers model name
                - 'all-MiniLM-L6-v2': 384-D, fast, recommended
                - 'all-mpnet-base-v2': 768-D, more accurate, slower
        """
        print(f"=== Initializing Embedding Generator ===")
        print(f"Model: {model_name}")
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load Sentence-Transformers model"""
        print("Loading model...")
        start_time = time.time()
        
        self.model = SentenceTransformer(self.model_name)
        
        # Get embedding dimension
        test_embedding = self.model.encode(["test"])
        self.embedding_dim = test_embedding.shape[1]
        
        print(f"Model loaded in {time.time() - start_time:.2f}s")
        print(f"Embedding dimension: {self.embedding_dim}")
        
    def load_parquet(self, parquet_path):
        """
        Load Parquet file from ETL output
        
        Returns:
            df: Pandas DataFrame
        """
        print(f"\n=== Loading Parquet File ===")
        print(f"Path: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        
        print(f"Loaded {len(df):,} movies")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check required fields
        if 'soup' not in df.columns:
            raise ValueError("Missing 'soup' column in DataFrame!")
        
        # Remove null soup
        null_count = df['soup'].isnull().sum()
        if null_count > 0:
            print(f"Warning: Removing {null_count} movies with null soup")
            df = df[df['soup'].notnull()]
        
        return df
    
    def generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings in batches
        
        Args:
            texts: List of texts
            batch_size: Batch processing size
            
        Returns:
            embeddings: numpy array of shape (n, embedding_dim)
        """
        print(f"\n=== Generating Embeddings ===")
        print(f"Total texts: {len(texts):,}")
        print(f"Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Use tqdm for progress display
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"Generation complete in {elapsed_time:.2f}s")
        print(f"Speed: {len(texts)/elapsed_time:.1f} texts/second")
        
        return embeddings
    
    def add_embeddings_to_df(self, df, embeddings):
        """
        Add embeddings to DataFrame
        
        Args:
            df: Pandas DataFrame
            embeddings: numpy array
            
        Returns:
            df: Updated DataFrame
        """
        print(f"\n=== Adding Embeddings to DataFrame ===")
        
        # Convert numpy array to list (for storage)
        df['embedding'] = embeddings.tolist()
        
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Sample embedding: {embeddings[0][:5]}... (showing first 5 dims)")
        
        return df
    
    def save_to_parquet(self, df, output_path):
        """
        Save DataFrame with embeddings as Parquet
        
        Args:
            df: DataFrame with embeddings
            output_path: Output path
        """
        print(f"\n=== Saving to Parquet ===")
        print(f"Output: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        print("Save complete!")
    
    def run_pipeline(self, input_parquet, output_parquet, batch_size=32):
        """
        Run complete embedding generation pipeline
        
        Args:
            input_parquet: ETL output parquet file
            output_parquet: Output file with embeddings
            batch_size: Batch processing size
        """
        start_time = time.time()
        
        # 1. Load model
        self.load_model()
        
        # 2. Load data
        df = self.load_parquet(input_parquet)
        
        # 3. Generate embeddings
        embeddings = self.generate_embeddings(
            df['soup'].tolist(),
            batch_size=batch_size
        )
        
        # 4. Add to DataFrame
        df = self.add_embeddings_to_df(df, embeddings)
        
        # 5. Save results
        self.save_to_parquet(df, output_parquet)
        
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Embedding Pipeline Complete!")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Movies processed: {len(df):,}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Output: {output_parquet}")
        print(f"{'='*50}")
        
        return df


def estimate_processing_time(num_movies, texts_per_second=100):
    """
    Estimate processing time
    
    Args:
        num_movies: Number of movies
        texts_per_second: Texts processed per second (varies by hardware)
    """
    estimated_seconds = num_movies / texts_per_second
    estimated_minutes = estimated_seconds / 60
    
    print(f"Estimated processing time: {estimated_minutes:.1f} minutes")
    print(f"(Based on {texts_per_second} texts/second)")


if __name__ == "__main__":
    # Configuration
    INPUT_PARQUET = './data/preprocessed/movie_soup.parquet'
    OUTPUT_PARQUET = './data/preprocessed/movies_with_embeddings.parquet'
    
    # Model selection
    # 'all-MiniLM-L6-v2': 384-D, fast (~100 texts/sec on CPU)
    # 'all-mpnet-base-v2': 768-D, slow (~30 texts/sec on CPU)
    MODEL_NAME = 'all-MiniLM-L6-v2'
    
    BATCH_SIZE = 32  # Adjust based on memory (16-64)
    
    # Run
    generator = EmbeddingGenerator(model_name=MODEL_NAME)
    df = generator.run_pipeline(
        input_parquet=INPUT_PARQUET,
        output_parquet=OUTPUT_PARQUET,
        batch_size=BATCH_SIZE
    )
    
    # Show samples
    print("\n=== Sample Data ===")
    print(df[['title', 'year', 'num_ratings']].head())
    print(f"\nEmbedding sample (first movie, first 10 dims):")
    print(df['embedding'].iloc[0][:10])
