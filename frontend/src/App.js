import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [kbFiles, setKbFiles] = useState([]);
  const [filteredFiles, setFilteredFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [isStreaming, setIsStreaming] = useState(true);
  const [bookmarkedFiles, setBookmarkedFiles] = useState([]);
  
  // Use ref for event source to properly clean up
  const eventSourceRef = useRef(null);

  // API URL - replace with your actual backend URL
  const API_URL = 'http://localhost:8000';

  // Fetch KB files on component mount
  useEffect(() => {
    fetchKbFiles();
    
    // Load bookmarked files from localStorage
    const savedBookmarks = localStorage.getItem('bookmarkedFiles');
    if (savedBookmarks) {
      setBookmarkedFiles(JSON.parse(savedBookmarks));
    }
    
    // Clean up event source when component unmounts
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Helper function to remove .txt extension for display
  const displayFileName = (filename) => {
    return filename.endsWith('.txt') ? filename.slice(0, -4) : filename;
  };

  // Toggle bookmark for a file
  const toggleBookmark = (filename, e) => {
    // Prevent triggering file selection when clicking bookmark icon
    e.stopPropagation();
    
    let updatedBookmarks;
    if (bookmarkedFiles.includes(filename)) {
      // Remove from bookmarks
      updatedBookmarks = bookmarkedFiles.filter(file => file !== filename);
    } else {
      // Add to bookmarks
      updatedBookmarks = [...bookmarkedFiles, filename];
    }
    
    // Update state and save to localStorage
    setBookmarkedFiles(updatedBookmarks);
    localStorage.setItem('bookmarkedFiles', JSON.stringify(updatedBookmarks));
  };

  // Fetch KB files from backend
  const fetchKbFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/kb`);
      const data = await response.json();
      const files = data.files || [];
      setKbFiles(files);
      
      // For initial display, prioritize bookmarked files
      const bookmarked = files.filter(file => bookmarkedFiles.includes(file.name));
      const nonBookmarked = files.filter(file => !bookmarkedFiles.includes(file.name));
      
      let initialFiles;
      if (bookmarked.length >= 4) {
        initialFiles = bookmarked.slice(0, 4);
      } else {
        initialFiles = [...bookmarked, ...nonBookmarked.slice(0, 4 - bookmarked.length)];
      }
      
      setFilteredFiles(initialFiles);
      setError('');
    } catch (err) {
      setError('Failed to fetch knowledge base files. Please try again later.');
      console.error('Error fetching KB files:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle search input
  const handleSearch = (e) => {
    const term = e.target.value.toLowerCase();
    setSearchTerm(term);
    
    let filtered;
    
    if (!term.trim()) {
      // When no search term, prioritize bookmarked files, then show remaining to fill up to 4
      const bookmarked = kbFiles.filter(file => bookmarkedFiles.includes(file.name));
      const nonBookmarked = kbFiles.filter(file => !bookmarkedFiles.includes(file.name));
      
      if (bookmarked.length >= 4) {
        filtered = bookmarked.slice(0, 4);
      } else {
        filtered = [...bookmarked, ...nonBookmarked.slice(0, 4 - bookmarked.length)];
      }
    } else {
      // Filter files based on search term
      const allMatched = kbFiles.filter(file => 
        displayFileName(file.name).toLowerCase().includes(term) || file.name.toLowerCase().includes(term)
      );
      
      // Sort matched files to prioritize bookmarked ones
      allMatched.sort((a, b) => {
        const aBookmarked = bookmarkedFiles.includes(a.name);
        const bBookmarked = bookmarkedFiles.includes(b.name);
        
        if (aBookmarked && !bBookmarked) return -1;
        if (!aBookmarked && bBookmarked) return 1;
        return 0;
      });
      
      // Limit to max 4 results even when searching
      filtered = allMatched.slice(0, 4);
    }
    
    setFilteredFiles(filtered);
  };

  // Handle file selection
  const handleFileSelect = async (filename) => {
    setLoading(true);
    setSelectedFile(filename);
    setAnswer('');
    
    try {
      // Fetch the file content - use full filename for backend
      const response = await fetch(`${API_URL}/kb/${filename}`);
      const data = await response.json();
      
      // Process the file for question answering
      const formData = new FormData();
      const file = new File([data.content], filename, { type: 'text/plain' });
      formData.append('file', file);
      
      setIsProcessing(true);
      const processResponse = await fetch(`${API_URL}/process`, {
        method: 'POST',
        body: formData,
      });
      
      const processData = await processResponse.json();
      setSessionId(processData.session_id);
      setError('');
    } catch (err) {
      setError(`Failed to load file "${displayFileName(filename)}". Please try again.`);
      console.error('Error loading file:', err);
    } finally {
      setLoading(false);
      setIsProcessing(false);
    }
  };

  // Handle asking a question with streaming response
  const handleAskQuestionStream = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    
    setIsProcessing(true);
    setAnswer('');
    
    try {
      // Use fetch with streaming response
      const response = await fetch(`${API_URL}/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      // Get the response reader
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Process the stream
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        // Decode and process the chunk
        const chunk = decoder.decode(value, { stream: true });
        
        // Parse line by line
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              if (data.text) {
                setAnswer(prevAnswer => prevAnswer + data.text);
              } else if (data.error) {
                setError(data.error);
              }
            } catch (error) {
              console.error("Error parsing stream data:", error);
            }
          }
        }
      }
    } catch (err) {
      setError(`Failed to get streaming response: ${err.message}`);
      console.error('Streaming error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle asking a question (non-streaming)
  const handleAskQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    
    setIsProcessing(true);
    setAnswer('');
    
    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      
      const data = await response.json();
      setAnswer(data.answer);
      setError('');
    } catch (err) {
      setError('Failed to get an answer. Please try again.');
      console.error('Error asking question:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Clear session and start a new search
  const handleClearSession = async () => {
    // Close any existing event source
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    
    setIsProcessing(true);
    
    try {
      await fetch(`${API_URL}/clear`, {
        method: 'POST',
      });
      
      // Reset state
      setSelectedFile(null);
      setQuestion('');
      setAnswer('');
      setSessionId('');
      setError('');
      setIsStreaming(true);
    } catch (err) {
      setError('Failed to clear session. Please try again.');
      console.error('Error clearing session:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Knowledge Base QA System</h1>
      </header>
      
      <main className="main">
        <div className="container">
          {error && <div className="error-message">{error}</div>}
          
          {/* Action buttons moved to Ask a Question area */}
          
          <div className="sections">
            <div className="kb-section">
              <h2>Knowledge Base</h2>
              
              {/* Search and Knowledge Base Files List */}
              <div className="kb-files">
                <div className="search-container">
                  <input
                    type="text"
                    className="search-input"
                    placeholder="Search knowledge base files..."
                    value={searchTerm}
                    onChange={handleSearch}
                    disabled={loading || isProcessing}
                  />
                  {searchTerm && (
                    <button 
                      onClick={() => {
                        setSearchTerm('');
                        handleSearch({target: {value: ''}});
                      }} 
                      className="btn btn-clear"
                      disabled={loading}
                    >
                      Clear
                    </button>
                  )}
                  <button 
                    onClick={fetchKbFiles} 
                    className="btn btn-refresh"
                    disabled={loading}
                  >
                    {loading ? 'Refreshing...' : 'Refresh'}
                  </button>
                </div>
                
                {loading ? (
                  <p className="loading-text">Loading knowledge base files...</p>
                ) : filteredFiles.length > 0 ? (
                  <div>
                    <ul className="file-list">
                      {filteredFiles.map((file) => (
                        <li 
                          key={file.name}
                          className={selectedFile === file.name ? 'selected' : ''}
                          onClick={() => handleFileSelect(file.name)}
                        >
                          <div className="file-info">
                            <span className="file-name">{displayFileName(file.name)}</span>
                            <span className="file-size">({formatFileSize(file.size)})</span>
                          </div>
                          <button 
                            className={`bookmark-btn ${bookmarkedFiles.includes(file.name) ? 'bookmarked' : ''}`}
                            onClick={(e) => toggleBookmark(file.name, e)}
                            title={bookmarkedFiles.includes(file.name) ? "Remove bookmark" : "Bookmark this file"}
                          >
                            {bookmarkedFiles.includes(file.name) ? '★' : '☆'}
                          </button>
                        </li>
                      ))}
                    </ul>
                    {!searchTerm.trim() && kbFiles.length > 4 && (
                      <p className="kb-info">
                        Showing 4 of {kbFiles.length} knowledge base files. Use search to find more.
                      </p>
                    )}
                    {searchTerm.trim() && (
                      <p className="kb-info">
                        {(() => {
                          const totalMatches = kbFiles.filter(file => 
                            displayFileName(file.name).toLowerCase().includes(searchTerm.toLowerCase()) || 
                            file.name.toLowerCase().includes(searchTerm.toLowerCase())
                          ).length;
                          
                          if (totalMatches > 4) {
                            return `Showing 4 of ${totalMatches} matching files. Refine your search for different results.`;
                          } else if (totalMatches > 0) {
                            return `Showing all ${totalMatches} matching files.`;
                          }
                          return '';
                        })()}
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="no-files-text">
                    {kbFiles.length > 0 
                      ? "No files match your search criteria." 
                      : "No files found in the knowledge base."}
                  </p>
                )}
              </div>
            </div>
            
            <div className="content-section">
              {selectedFile ? (
                <>
                  <div className="content-header">
                    <h2>Ask a Question about <span className="highlight-file">{displayFileName(selectedFile)}</span></h2>
                    <button 
                      onClick={handleClearSession} 
                      className="btn btn-secondary"
                      disabled={isProcessing || !sessionId}
                    >
                      Start New Search
                    </button>
                  </div>
                  
                  <div className="qa-section">
                    <form onSubmit={isStreaming ? handleAskQuestionStream : handleAskQuestion} className="question-form">
                      <div className="input-with-icon">
                        <i className="search-icon">🔍</i>
                        <input
                          type="text"
                          value={question}
                          onChange={(e) => setQuestion(e.target.value)}
                          placeholder="What would you like to know?"
                          disabled={isProcessing}
                          className="question-input"
                        />
                        <button
                          type="submit"
                          className="btn btn-primary"
                          disabled={!question.trim() || isProcessing}
                        >
                          {isProcessing ? 'Thinking...' : 'Search'}
                        </button>
                      </div>
                      
                      <div className="stream-option">
                        <label className="checkbox">
                          <input
                            type="checkbox"
                            checked={isStreaming}
                            onChange={(e) => setIsStreaming(e.target.checked)}
                            disabled={isProcessing}
                          />
                          <span>Enable real-time streaming response</span>
                        </label>
                      </div>
                    </form>
                    
                    {answer && (
                      <div className={`answer ${isProcessing && isStreaming ? 'streaming' : ''}`}>
                        <h4>Answer</h4>
                        <div className="answer-content">
                          {answer}
                        </div>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="no-selection">
                  <p>Select a file from the knowledge base to ask questions.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
      
      <footer className="footer">
        <p>Knowledge Base QA System &copy; 2025</p>
        <p>Developed by Sukhpreet Singh</p>
      </footer>
    </div>
  );
}

// Helper function to format file size
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default App;