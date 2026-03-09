from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the class wrote in search_movies.py file
from search_movies import MovieSearchEngine 

app = FastAPI()

# ── CORS Setup ──
# This is required so your React app (on port 5173) 
# is allowed to get data from this Python script (on port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the search engine once when the server starts
# This loads the model and connects to your Postgres/S3 database
engine = MovieSearchEngine()

@app.get("/api/search")
async def search_vibe(vibe: str = Query(...)):
    """
    This endpoint receives the 'vibe' from React and 
    returns the top 5 movie matches.
    """
    # Calls search function
    raw_results = engine.search_movies(vibe, top_k=5)
    
    # Format the database results into a clean list for the website
    formatted_movies = []
    for result in raw_results:
        # result matches the SELECT order in search_movies.py
        formatted_movies.append({
            "id": result[0],
            "title": result[1],
            "year": result[2],
            "genres": result[3],
            "rating": float(result[4]),
            "distance": float(result[6]),
            "similarity": round(float(1 - result[6]), 3) # Convert distance to a 0-1 score
        })
    
    return {"movies": formatted_movies}

if __name__ == "__main__":
    # This starts the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)