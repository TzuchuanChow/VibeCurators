import React from 'react';
import './App.css';

function App() {
  return (
    <div className="min-h-screen bg-linear-to-t from-sky-500 to-indigo-500 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl text-center">
        {/* Title */}
        <h1 className="font-mono text-8xl font-bold mb-8 tracking-tight text-slate-900">
          VibeLens
        </h1>

        {/* Search Bar Container */}
        <div className="relative group">
          <input
            type="text"
            placeholder="Search your vibe..."
            className="w-full px-6 py-4 text-lg rounded-full border border-slate-200 shadow-sm 
                       outline-none focus:ring-2 focus:ring-indigo-600 focus:border-transparent 
                       transition-all duration-300 bg-white"
          />
          <button className="absolute right-4 top-1/2 -translate-y-1/2 bg-slate-900 text-white px-6 py-2 rounded-full hover:bg-blue-800 transition-colors">
            Search
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;