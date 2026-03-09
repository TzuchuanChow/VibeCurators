import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom'; 
import './App.css';

function App() {
  {/* State to hold the vibe input by the user and navigation function */}
  const [vibe, setVibe] = useState('');
  const navigate = useNavigate();

  {/* When the user clicks the search button, navigate to the /movies page with the vibe as a query parameter */}
  const handleSearch = () => {
    if (vibe.trim()) {
      navigate(`/movies?vibe=${encodeURIComponent(vibe)}`);
    }
  };
  return (
    <div className="min-h-screen bg-linear-to-t from-indigo-800 to-sky-700 flex flex-col justify-between p-4">
      {/* Center Section of the Web Page*/}
      <div className="flex flex-col items-center justify-center grow">
        <div className="w-full max-w-2xl text-center">

          {/* Title */}
          <h1 className="font-mono text-8xl font-bold mb-8 tracking-tight text-black">
            VibeLens
          </h1>
          {/* Subtitle */}
          <p className="text-slate-200 text-lg mb-6 font-mono">
            Discover movies by vibe, not just titles
          </p>

          {/* Search Bar */}
          <div className="relative group">
            <input
              type="text"
              placeholder="Try: dark sci-fi about time loops..."
              // getting the vibe from the input and storing it in state
              value={vibe}
              onChange={(e) => setVibe(e.target.value)}
              // Styling for the input field with focus
              className="w-full px-6 py-4 text-lg rounded-full border border-slate-200 shadow-sm 
                         outline-none focus:ring-2 focus:ring-indigo-600 focus:border-transparent 
                        bg-white"
            />
            {/* Search Button */}
            <button 
              onClick={handleSearch}
              className="absolute right-4 top-1/2 -translate-y-1/2 bg-slate-900 text-white px-6 py-2 rounded-full hover:bg-blue-800 cursor-pointer"
            >
              Search
            </button>
          </div>

        </div>
      </div>

      {/* Footer */}
      <footer className="text-center pb-4">
        <p className="text-sm text-slate-200 font-mono">
          VibeLens — Semantic Movie Search <br />
          Big Data Management — Winter 2026 | The Vibe Curators Team
        </p>
      </footer>

    </div>
  );
}

export default App;