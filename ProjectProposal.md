## Vibe Curators group project ##

Big Data - CS226, Winter 2026

### Abstract
The rapid rise of movies has created challenges for the tradi-
tional movie recommendation system to deliver accurate results
based on the user mood or specific plot points to users. The goal
of this project is to develop a semantic movie recommendation
system using movie datasets such as IMDB, where users can search
movies based on descriptive queries or "vibe" rather than just genres
or ratings. By implementing vector database and RAG(Retrieval-
Augmented Generation), the recommendation system can process
large complex unstructured data efficiently. This project provides
hands-on experience with big data management tools, and creating
an interactive search system for users based on their preference.


### 1 Problem Background
In today’s world, video entertainment has grown rapidly, re-
sulting in having numerous amounts of content across multiple
platforms, making movie search a significant challenge for users. A
traditional recommendation system typically relies on filters such
as genres, ratings, or release years, but it fails to capture the users
current mood or specific movie plot points. For example, if a user
searches "dark sci-fi about time loops" the system will struggle to
return the best accurate movie that matches, because most of the
search feature depends on categories rather than semantic meaning
of the query. This creates a gap between what the user prefers and
what a typical recommendation system can deliver.
As movie datasets continue to grow in both size and complexity,
this problem makes it more difficult for using data management
approaches. For example of a large dataset, IMDB contains millions
of records with unstructured data like images, titles, reviews, or year.
Managing and creating queries efficiently with these type of data is
challenging for creating recommendations methods. By addressing
this problem requires modern data management methods that can
scale and handle complex data.


### 2 Motivation
The motivation for this project is to make the movie selection
process easier for users, as it can be a time consuming process. In
today’s day and age, it can be difficult to select movies especially
based just on title. We will leverage the descriptions of movies to
allow users to search by "vibe" and make their final selection based
on reviews.
We chose PostgreSQL for our project because it is great for
advanced work, SQL queries, and provides the correct amount of
storage for our datasets. We chose this over MongoDB Atlas because
of the storage size, and over AWS database because PostgreSQL is
free.


### 3 Datasets
We will use a hybrid dataset combining the MovieLens dataset
and the TMDB (The Movie Database) metadata. MovieLens pro-
vides high-quality user ratings and tags, while TMDB provides the
essential "plot overviews" and "user reviews" required for seman-
tic vector search. We will use the links.csv mapping file to join
these datasets by their respective IMDB and TMDB IDs, creating
an unified "Movie Soup" of text for our RAG pipeline.

### 4 Project Outcome
The final outcome will be an Intelligent Semantic Movie Naviga-
tor. Unlike traditional systems that filter by genre, this system will
provide a chat interface where users can describe complex "vibes"
(e.g., "movies about isolated researchers finding something mysteri-
ous in the ice"). The system will return a ranked list of movies and,
using RAG, will generate a natural language explanation for why
each movie matches the user’s specific "mood" or request.

### 5 Relevance to Big Data
This project addresses three core "Big Data" challenges: High-
Dimensional Data Management: We will manage 10,000+ movie
embeddings in a distributed Vector DBMS (like PostgreSQL) Index-
ing and Latency: We will implement Approximate Nearest Neigh-
bor (ANN) indexing (such as HNSW) to ensure sub-second re-
trieval times across a massive search space. Data Pipeline Com-
plexity: The project involves a multi-stage pipeline: ingestion of
raw TSV/JSON data, batch embedding generation via an LLM, and
real-time retrieval-augmentation.

### 6 Evaluation
We will evaluate the project using two primary metrics: Re-
trieval Accuracy. We will test the system with 20 "niche"
queries (e.g., "retro-futuristic heist movies") and measure how many
relevant titles appear in the top 10 results compared to a standard
keyword search. System Latency: We will benchmark the time taken
for the vector search vs. the RAG generation to ensure the "Big
Data" backend is performing efficiently under load.


### 7 Milestones and Proposed Timeline
• Concept & Setup, 1/21-1/29: Finalize dataset and system
design; Submit Proposal Presentation.
• Research & Backend Develop, 1/29-2/19: Implement Data
Ingestion and Vectorization pipeline; Complete Literature
Survey.
• Integration & Optimization, 2/19-2/26: Connect RAG
logic with PostgreSQL; Submit Report Outline.
• Front-end & Testing, 2/26-3/5: Build Chat UI; Draft initial
report and conduct latency & accuracy tests.
• Final Delivery, 3/5-3/17: Final Report, Product Demo, and
Final Presentation.


### 8 Bibliography
[1] Asaniczka (2026) Full TMDB movies dataset 2024 (1M movies),
Kaggle. Available at: https://www.kaggle.com/ datasets/asaniczka/
tmdb-movies-dataset-2023-930k-movies (Accessed: 21 January 2026).
[2] Group, P.G.D. (2026) PostgreSQL. Available at: https://www.post
gresql.org/ (Accessed: 21 January 2026).
[3] MongoDB atlas (2026) MongoDB. Available at: https://www.mon
godb.com/cloud/atlas/register (Accessed: 21 January 2026).
[4] Movielens 1M Dataset (2021) GroupLens. Available at: https://gro
uplens.org/datasets/movielens/1m/ (Accessed: 21 January 2026).
