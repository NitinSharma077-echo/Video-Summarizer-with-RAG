import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Bot,
  CheckCircle,
  Download,
  FileDown,
  FileText,
  Link,
  List,
  LoaderCircle,
  MessageSquare,
  Send,
  UploadCloud,
  Video,
  X,
} from 'lucide-react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || '/api';

function App() {
  const [mode, setMode] = useState('upload');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [chatQuery, setChatQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const resetResult = () => {
    setData(null);
    setChatHistory([]);
    setError('');
  };

  const handleFiles = (files) => {
    const selected = files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(selected);
    });
    resetResult();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    handleFiles(event.dataTransfer.files);
  };

  const handleProcess = async (event) => {
    event.preventDefault();
    setError('');

    if (mode === 'upload' && !file) {
      setError('Choose a video or audio file first.');
      return;
    }

    if (mode === 'url' && !url.trim()) {
      setError('Paste a video URL first.');
      return;
    }

    setLoading(true);
    try {
      const response =
        mode === 'upload'
          ? await uploadFile(file)
          : await axios.post(`${API_URL}/process`, { url: url.trim() });
      setData(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || 'Could not process this video.');
    } finally {
      setLoading(false);
    }
  };

  const uploadFile = (selectedFile) => {
    const formData = new FormData();
    formData.append('file', selectedFile);
    return axios.post(`${API_URL}/upload`, formData);
  };

  const handleChat = async (event) => {
    event.preventDefault();
    if (!chatQuery.trim()) return;

    const question = chatQuery.trim();
    setChatQuery('');
    setChatHistory((history) => [...history, { role: 'user', content: question }]);
    setChatLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, { query: question });
      setChatHistory((history) => [...history, { role: 'bot', content: response.data.answer }]);
    } catch (err) {
      console.error(err);
      setChatHistory((history) => [
        ...history,
        { role: 'bot', content: 'Sorry, I could not answer from the processed video.' },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="header">
        <div className="brand">
          <Video className="brand-icon" />
          <div>
            <h1>Video Insight Studio</h1>
            <p>Summarize, extract decisions, and chat with any video or audio file.</p>
          </div>
        </div>
      </header>

      <main className="main-content">
        {!data && (
          <section className="input-panel">
            <div className="mode-tabs" role="tablist" aria-label="Processing mode">
              <button
                type="button"
                className={mode === 'upload' ? 'active' : ''}
                onClick={() => {
                  setMode('upload');
                  resetResult();
                }}
              >
                <UploadCloud size={18} />
                Upload file
              </button>
              <button
                type="button"
                className={mode === 'url' ? 'active' : ''}
                onClick={() => {
                  setMode('url');
                  resetResult();
                }}
              >
                <Link size={18} />
                Video URL
              </button>
            </div>

            <form onSubmit={handleProcess} className="processor-form">
              {mode === 'upload' ? (
                <label
                  className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={(event) => {
                    event.preventDefault();
                    setDragActive(true);
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={() => setDragActive(false)}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    accept="video/*,audio/*"
                    onChange={(event) => handleFiles(event.target.files)}
                  />
                  <UploadCloud className="drop-icon" />
                  <span className="drop-title">
                    {file ? file.name : 'Drop a video here or click to browse'}
                  </span>
                  <span className="drop-meta">
                    {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'MP4, MOV, MKV, MP3, WAV and more'}
                  </span>
                </label>
              ) : (
                <div className="url-field">
                  <Link size={20} />
                  <input
                    type="url"
                    placeholder="Paste any public video URL..."
                    value={url}
                    onChange={(event) => {
                      setUrl(event.target.value);
                      resetResult();
                    }}
                  />
                </div>
              )}

              {file && mode === 'upload' && (
                <button type="button" className="clear-file" onClick={() => setFile(null)}>
                  <X size={16} />
                  Remove selected file
                </button>
              )}

              {error && <p className="error-message">{error}</p>}

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? <LoaderCircle className="spin" size={20} /> : <Video size={20} />}
                {loading ? 'Processing...' : 'Analyze video'}
              </button>
            </form>
          </section>
        )}

        {loading && (
          <section className="loading-state">
            <LoaderCircle className="loading-icon spin" />
            <p>Extracting audio, transcribing, and building your video knowledge base...</p>
          </section>
        )}

        {data && (
          <section className="results-layout">
            <div className="results-main">
              <div className="results-header">
                <div>
                  <span className="eyebrow">Analysis complete</span>
                  <h2>{data.title || 'Video analysis results'}</h2>
                </div>
                <button type="button" onClick={resetResult} className="secondary-btn">
                  Analyze another
                </button>
                <a href="#video-chat" className="secondary-btn">
                  Ask the video
                </a>
                <a
                  href={`${API_URL}/download/transcript`}
                  className="secondary-btn"
                  download
                  title="Download the transcript, summary, and highlights as a PDF"
                >
                  <FileDown size={16} />
                  Download transcript (PDF)
                </a>
                {data.media_available && (
                  <a
                    href={`${API_URL}/download/media`}
                    className="secondary-btn"
                    download
                    title="Download the original video/audio file"
                  >
                    <Download size={16} />
                    Download video/audio
                  </a>
                )}
              </div>

              {previewUrl && mode === 'upload' && (
                <video className="video-preview" src={previewUrl} controls />
              )}

              <ResultSection icon={FileText} title="Summary" content={data.summary} />
              <ResultSection icon={List} title="Key Points" content={data.key_points} />
              <ResultSection icon={CheckCircle} title="Action Items" content={data.action_items} />
              <ResultSection icon={Video} title="Key Moments" content={data.key_moments} />
            </div>

            <aside className="chat-panel" id="video-chat">
              <div className="chat-header">
                <Bot className="chat-icon" />
                <div>
                  <h3>Ask the video</h3>
                  <p>Answers are based on the processed transcript.</p>
                </div>
              </div>
              <div className="chat-history">
                {chatHistory.length === 0 && (
                  <div className="empty-chat">
                    <MessageSquare size={28} />
                    <span>Ask for examples, decisions, dates, or a simpler explanation.</span>
                  </div>
                )}
                {chatHistory.map((msg, index) => (
                  <div key={`${msg.role}-${index}`} className={`chat-message ${msg.role}`}>
                    {msg.content}
                  </div>
                ))}
                {chatLoading && <div className="chat-message bot">Thinking...</div>}
              </div>
              <form className="chat-form" onSubmit={handleChat}>
                <input
                  type="text"
                  placeholder="Ask about the video..."
                  value={chatQuery}
                  onChange={(event) => setChatQuery(event.target.value)}
                />
                <button type="submit" disabled={chatLoading}>
                  <Send size={18} />
                </button>
              </form>
            </aside>
          </section>
        )}
      </main>
    </div>
  );
}

function ResultSection({ icon: Icon, title, content }) {
  return (
    <article className="result-section">
      <h3>
        <Icon size={20} />
        {title}
      </h3>
      <pre>{content || 'No content returned.'}</pre>
    </article>
  );
}

export default App;
