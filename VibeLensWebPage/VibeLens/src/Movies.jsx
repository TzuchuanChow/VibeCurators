import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';

function Movies() {
  // State to hold the vibe from URL, movie results, and loading status
  const [searchParams] = useSearchParams();
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  
  // Get the "vibe" from the URL
  const vibe = searchParams.get("vibe");

  useEffect(() => {
    const fetchMovies = async () => {
      setLoading(true);
      try {
        // Fetching data from your Python API on port 8000
        const response = await fetch(`http://127.0.0.1:8000/api/search?vibe=${encodeURIComponent(vibe)}`);
        const data = await response.json();
        
        // Store the movie list in our state
        setMovies(data.movies);
      } catch (error) {
        // Display an error message if the API call fails else it will continue loading
        console.error("Is the Python API running? Error:", error);
      } finally {
        setLoading(false);
      }
    };

    if (vibe) fetchMovies();
  }, [vibe]);

  return (
    <div className="min-h-screen bg-linear-to-t from-indigo-800 to-sky-700 text-white p-8 font-sans">
      {/* Header */}
      <div className="flex justify-between items-center mb-12">
        {/* Back Button and Title */}
        <button 
          onClick={() => navigate('/')}
          className="bg-slate-900 text-white hover:bg-blue-800 px-4 py-2 rounded-full cursor-pointer"
        >
          New Search? 
        </button>
        <h1 className="text-xl font-mono font-bold tracking-tighter text-black">VibeLens</h1>
      </div>

      {/* Search Result That User Typed  */}
      <div className="text-center mb-10">
        <p className="text-violet-300 font-mono text-lg uppercase tracking-widest shadow-2xl">Showing results for</p>
        <h2 className="text-4xl font-bold italic text-white mt-2">"{vibe}"</h2>
      </div>

      {/* Movie Results List */}
      <div className="max-w-5xl mx-auto grid gap-6">
        {/* Show loading state */}
        {loading ? (
          <div className="flex flex-col items-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mb-4"></div>
            <p className="text-slate-400 font-mono">Consulting the VibeLens Engine...</p>
          </div>
        ) : (
          movies.map((movie) => (
            <div
              // Design movie card 
              key={movie.id} 
              className="group bg-slate-800/50 border border-indigo-500 p-6 rounded-2xl flex justify-between items-center hover:bg-slate-800 hover:border-indigo-500 shadow-xl"
            >
              <div className="flex-1">
                {/* Display Movie title and year from the API */}
                <h3 className="text-2xl font-bold group-hover:text-indigo-400 transition-colors">
                  {movie.title} <span className="font-normal ml-2">({movie.year})</span>
                </h3>
                {/* Display Movie genres from the API */}
                <p className="text-slate-300 mt-1">{movie.genres}</p>
              </div>
              
              <div className="flex flex-col items-end ml-8">
                {/* Display similarity score */}
                <div className="bg-indigo-900/30 text-indigo-300 px-4 py-1 rounded-full text-xs font-mono border border-indigo-500/30 mb-2">
                  <p>{Math.round(movie.similarity * 100)}% Vibe Match</p>
                </div>
                {/* Display Movie rating from the API */}
                <div className="text-yellow-500 font-bold text-lg">
                  <p>{movie.rating.toFixed(1)} <span className="text-xs text-yellow-600">⭐</span></p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <footer className="text-center pt-10">
        <p className="text-sm text-slate-200 font-mono">
          VibeLens — Semantic Movie Search <br />
          Big Data Management — Winter 2026 | The Vibe Curators Team
        </p>
      </footer>
    </div>
  );
}

export default Movies;