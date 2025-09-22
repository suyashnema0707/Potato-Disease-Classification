// In src/App.jsx
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.png', '.jpg'] },
    multiple: false,
  });

  const handleClassify = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error during classification:", error);
      alert("Failed to classify image. Ensure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="bg-gray-900 min-h-screen flex flex-col items-center justify-center text-white font-sans p-4">
      <div className="w-full max-w-2xl bg-gray-800 rounded-2xl shadow-2xl p-8 space-y-6">
        <header className="text-center">
          <h1 className="text-4xl font-bold text-emerald-400">Potato Leaf Disease Classifier</h1>
          <p className="text-gray-400 mt-2">Upload an image to classify its health.</p>
        </header>
        <div 
          {...getRootProps()} 
          className={`border-4 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all ${isDragActive ? 'border-emerald-500 bg-gray-700' : 'border-gray-600 hover:border-emerald-400'}`}
        >
          <input {...getInputProps()} />
          {preview ? (
            <img src={preview} alt="Preview" className="mx-auto max-h-60 rounded-lg" />
          ) : (
            <p>{isDragActive ? "Drop the image here..." : "Drag & drop an image, or click to select"}</p>
          )}
        </div>
        <div className="flex items-center justify-center space-x-4">
          <button onClick={handleClassify} disabled={!selectedFile || isLoading} className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-500 font-bold py-3 px-6 rounded-lg transition-all w-40">
            {isLoading ? <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mx-auto"></div> : "Classify Leaf"}
          </button>
          {preview && <button onClick={handleClear} className="bg-red-600 hover:bg-red-700 font-bold py-3 px-6 rounded-lg transition-all">Clear</button>}
        </div>
        {result && (
          <div className="bg-gray-700 p-6 rounded-lg text-center animate-fade-in">
            <h2 className="text-2xl font-semibold text-emerald-300">Result</h2>
            <p className="text-xl mt-2">Class: <span className="font-bold">{result.class.replace(/___/g, ' - ').replace(/_/g, ' ')}</span></p>
            <p className="text-lg mt-1">Confidence: <span className="font-bold">{(result.confidence * 100).toFixed(2)}%</span></p>
          </div>
        )}
      </div>
    </div>
  );
}
export default App;