"""
VibeLens ETL Pipeline - Spark Version
Process MovieLens + TMDB data and generate Movie Soup
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, concat_ws, lit, when, array_join,
    size, explode, trim, lower
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, ArrayType, LongType
)
import json
import time

class VibeLensETL:
    def __init__(self, config):
        """
        Initialize Spark ETL Pipeline
        
        Args:
            config (dict): Configuration dictionary with paths and thresholds
        """
        self.config = config
        self.spark = None
        
    def initialize_spark(self):
        """Initialize Spark Session"""
        print("=== Initializing Spark Session ===")
        
        self.spark = SparkSession.builder \
            .appName("VibeLens-ETL") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.driver.maxResultSize", "4g") \
            .getOrCreate()
        
        # Set log level
        self.spark.sparkContext.setLogLevel("WARN")
        
        print(f"Spark Version: {self.spark.version}")
        print(f"Spark UI: {self.spark.sparkContext.uiWebUrl}")
        
    def load_movielens_data(self):
        """
        Load MovieLens dataset
        Returns: (movies_df, ratings_df, links_df)
        """
        print("\n=== Loading MovieLens Data ===")
        
        # Load movies.csv
        print("Loading movies.csv...")
        movies_df = self.spark.read.csv(
            self.config['movielens_movies_path'],
            header=True,
            inferSchema=True
        )
        print(f"Movies count: {movies_df.count():,}")
        
        # Load ratings.csv (large file)
        print("Loading ratings.csv (900MB)...")
        start_time = time.time()
        ratings_df = self.spark.read.csv(
            self.config['movielens_ratings_path'],
            header=True,
            inferSchema=True
        )
        print(f"Ratings count: {ratings_df.count():,}")
        print(f"Load time: {time.time() - start_time:.2f}s")
        
        # Load links.csv
        print("Loading links.csv...")
        links_df = self.spark.read.csv(
            self.config['movielens_links_path'],
            header=True,
            inferSchema=True
        )
        print(f"Links count: {links_df.count():,}")
        
        return movies_df, ratings_df, links_df
    
    def load_tmdb_data(self):
        """
        Load TMDB JSON data
        Returns: tmdb_df
        """
        print("\n=== Loading TMDB Data ===")
        
        # TMDB data schema
        tmdb_schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("overview", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("genres", ArrayType(StructType([
                StructField("id", IntegerType()),
                StructField("name", StringType())
            ])), True),
            StructField("vote_average", FloatType(), True),
            StructField("runtime", IntegerType(), True),
            StructField("cast", ArrayType(StructType([
                StructField("name", StringType()),
                StructField("character", StringType())
            ])), True),
            StructField("crew", ArrayType(StructType([
                StructField("name", StringType()),
                StructField("job", StringType())
            ])), True),
            StructField("reviews", ArrayType(StructType([
                StructField("author", StringType()),
                StructField("content", StringType()),
                StructField("rating", FloatType())
            ])), True)
        ])
        
        print(f"Loading TMDB JSON from {self.config['tmdb_path']}...")
        tmdb_df = self.spark.read.json(
            self.config['tmdb_path'],
            schema=tmdb_schema,
            multiLine=True  # Support multiline JSON
        )
        
        print(f"TMDB movies count: {tmdb_df.count():,}")
        
        return tmdb_df
    
    def compute_rating_statistics(self, ratings_df):
        """
        Compute rating statistics for each movie
        
        Args:
            ratings_df: Ratings DataFrame
            
        Returns:
            rating_stats_df: Contains movieId, num_ratings, avg_rating
        """
        print("\n=== Computing Rating Statistics ===")
        
        rating_stats = ratings_df.groupBy("movieId").agg(
            count("*").alias("num_ratings"),
            avg("rating").alias("avg_rating")
        )
        
        print(f"Movies with ratings: {rating_stats.count():,}")
        
        # Apply rating threshold filter
        min_ratings = self.config.get('min_ratings_threshold', 100)
        filtered_stats = rating_stats.filter(col("num_ratings") >= min_ratings)
        
        print(f"Movies with >= {min_ratings} ratings: {filtered_stats.count():,}")
        
        return filtered_stats
    
    def create_movie_soup(self, movies_df, links_df, tmdb_df, rating_stats_df):
        """
        Merge all data sources and generate Movie Soup
        
        Returns:
            movie_soup_df: Contains movieId, title, year, soup, etc.
        """
        print("\n=== Creating Movie Soup ===")
        
        # Step 1: Merge MovieLens movies + rating stats
        print("Step 1: Merging movies with rating statistics...")
        movies_with_stats = movies_df.join(
            rating_stats_df,
            on="movieId",
            how="inner"
        )
        print(f"After rating filter: {movies_with_stats.count():,}")
        
        # Step 2: Join with links table (get tmdbId)
        print("Step 2: Joining with links table...")
        movies_with_links = movies_with_stats.join(
            links_df.select("movieId", "tmdbId"),
            on="movieId",
            how="inner"
        )
        print(f"After tmdbId join: {movies_with_links.count():,}")
        
        # Step 3: Process TMDB data
        print("Step 3: Processing TMDB data...")
        
        # Extract genres list as string
        tmdb_processed = tmdb_df.withColumn(
            "genres_str",
            array_join(col("genres.name"), ", ")
        )
        
        # Extract main cast (top 5)
        tmdb_processed = tmdb_processed.withColumn(
            "cast_str",
            array_join(col("cast.name"), ", ")
        )
        
        # Extract director
        tmdb_processed = tmdb_processed.withColumn(
            "director",
            when(
                size(col("crew")) > 0,
                array_join(
                    explode(col("crew")).filter(col("job") == "Director").select("name"),
                    ", "
                )
            ).otherwise(lit(""))
        )
        
        # Extract year
        tmdb_processed = tmdb_processed.withColumn(
            "year",
            when(
                col("release_date").isNotNull(),
                col("release_date").substr(1, 4).cast(IntegerType())
            ).otherwise(lit(None))
        )
        
        # Select required fields
        tmdb_clean = tmdb_processed.select(
            col("id").alias("tmdbId"),
            col("overview"),
            col("genres_str"),
            col("cast_str"),
            col("year").alias("tmdb_year"),
            col("vote_average")
        )
        
        # Step 4: Join with TMDB data
        print("Step 4: Joining with TMDB data...")
        full_data = movies_with_links.join(
            tmdb_clean,
            on="tmdbId",
            how="inner"
        )
        print(f"After TMDB join: {full_data.count():,}")
        
        # Step 5: Generate Movie Soup
        print("Step 5: Generating Movie Soup...")
        
        # Extract year from MovieLens title (format: "Movie Title (1999)")
        full_data = full_data.withColumn(
            "ml_year",
            when(
                col("title").contains("("),
                col("title").substr(-5, 4).cast(IntegerType())
            ).otherwise(lit(None))
        )
        
        # Clean title (remove year part)
        full_data = full_data.withColumn(
            "clean_title",
            when(
                col("title").contains("("),
                trim(col("title").substr(1, col("title").length() - 7))
            ).otherwise(col("title"))
        )
        
        # Use TMDB year, fallback to MovieLens year
        full_data = full_data.withColumn(
            "final_year",
            when(col("tmdb_year").isNotNull(), col("tmdb_year"))
            .otherwise(col("ml_year"))
        )
        
        # Create Movie Soup (lightweight natural language format)
        movie_soup_df = full_data.withColumn(
            "soup",
            concat_ws(" ",
                col("clean_title"),
                lit("is a"),
                col("genres_str"),
                lit("film."),
                col("overview"),
                when(col("cast_str").isNotNull(), 
                     concat_ws(" ", lit("Starring"), col("cast_str"), lit("."))).otherwise(lit(""))
            )
        )
        
        # Select final fields
        final_df = movie_soup_df.select(
            col("movieId"),
            col("clean_title").alias("title"),
            col("final_year").alias("year"),
            col("genres"),  # MovieLens genres
            col("genres_str").alias("tmdb_genres"),
            col("num_ratings"),
            col("avg_rating"),
            col("vote_average").alias("tmdb_rating"),
            col("soup")
        )
        
        # Filter out null soup entries
        final_df = final_df.filter(col("soup").isNotNull())
        
        print(f"Final movie count: {final_df.count():,}")
        
        return final_df
    
    def save_to_parquet(self, df, output_path):
        """
        Save as Parquet format
        
        Args:
            df: DataFrame to save
            output_path: Output path
        """
        print(f"\n=== Saving to Parquet: {output_path} ===")
        
        # Repartition to optimize file size (avoid too many small files)
        df = df.repartition(10)
        
        df.write.mode("overwrite").parquet(output_path)
        
        print("Save complete!")
        
    def run_pipeline(self):
        """Run complete ETL Pipeline"""
        start_time = time.time()
        
        try:
            # 1. Initialize Spark
            self.initialize_spark()
            
            # 2. Load data
            movies_df, ratings_df, links_df = self.load_movielens_data()
            tmdb_df = self.load_tmdb_data()
            
            # 3. Compute rating statistics
            rating_stats_df = self.compute_rating_statistics(ratings_df)
            
            # 4. Create Movie Soup
            movie_soup_df = self.create_movie_soup(
                movies_df, links_df, tmdb_df, rating_stats_df
            )
            
            # 5. Show samples
            print("\n=== Sample Movie Soup ===")
            movie_soup_df.select("title", "year", "soup").show(3, truncate=80)
            
            # 6. Save results
            self.save_to_parquet(
                movie_soup_df,
                self.config['output_parquet_path']
            )
            
            # 7. Save sample to CSV (for inspection)
            print(f"\n=== Saving sample to CSV ===")
            sample_df = movie_soup_df.limit(100)
            sample_df.toPandas().to_csv(
                self.config['output_csv_sample_path'],
                index=False
            )
            
            total_time = time.time() - start_time
            print(f"\n{'='*50}")
            print(f"ETL Pipeline Complete!")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
            print(f"Output: {self.config['output_parquet_path']}")
            print(f"{'='*50}")
            
        finally:
            if self.spark:
                self.spark.stop()


if __name__ == "__main__":
    # Configuration paths (modify based on actual setup)
    config = {
        # MovieLens data paths
        'movielens_movies_path': './data/raw/movielens/movies.csv',
        'movielens_ratings_path': './data/raw/movielens/ratings.csv',
        'movielens_links_path': './data/raw/movielens/links.csv',
        
        # TMDB data path
        'tmdb_path': './data/raw/tmdb/*.json',  # Supports wildcards
        
        # Output paths
        'output_parquet_path': './data/preprocessed/movie_soup.parquet',
        'output_csv_sample_path': './data/preprocessed/movie_soup_sample.csv',
        
        # Parameters
        'min_ratings_threshold': 100
    }
    
    # Run ETL
    etl = VibeLensETL(config)
    etl.run_pipeline()
