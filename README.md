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

How to run locally

# 1. Clone code
git clone https://github.com/YourUsername/VibeCurators.git
cd VibeCurators

# 2. Install dependencies
conda create -n VibeLens python=3.11
conda activate VibeLens
pip install -r requirements.txt

# 3. Configure database credentials(using  information in the section below)
**For .env file:**
PG_HOST=your-database-host.example.com
PG_PORT=5432
PG_DATABASE=vibelens
PG_USER=postgres
PG_PASSWORD=your-database-password

cp .env.example .env
vim .env
Remove everything already exists in the “.env” file and copy text in the section above into it

*Make sure Transformers (NOT Sentence Transformer) is on version 4.57.6
Use “conda list” to check your current version 
Pip install transformers=4.57.6

# 4. Run search
python search_movies.py




# 🎬 VibeLens Web Page

Welcome to the VibeLens Webpage! This project is a React-based movie search application built with Vite and Tailwind CSS.

## 🚀 Getting Started

Follow these steps to get the development environment running on your machine.

### 1. Prerequisites
Make sure you have **Node.js** installed (LTS version recommended).
- [Download Node.js](https://nodejs.org/)
- Check your version in the terminal: `node -v`

### 2. Installation
Clone the repository (if you haven't already) and install the necessary dependencies make sure you are in the VibeLens Folder:

```bash
# In Bash
# Install all packages listed in package.json
npm install

# Running the App
npm run dev
```
### Running The backend API
Follow the guide get the envrioment running provided above .

- Once you got it running and still inside the enviroment run "python -m pip install fastapi uvicorn" 
- Then start the backend server run:  "python api.py."
You are successful when you see: Uvicorn running on http://127.0.0.1:8000 (leave it as it dont click) and go back to the web and TEST!


Once started, click the link in your terminal (usually http://localhost:5173) to view the site


