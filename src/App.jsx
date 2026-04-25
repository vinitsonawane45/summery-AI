import React, { useState, useEffect } from 'react';
import { 
  FileText, Search, Link as LinkIcon, Key, Smile, 
  User, Moon, Sun, Loader2, LogOut
} from 'lucide-react';
import './index.css';

const API_BASE = ''; // Proxy handles this

function App() {
  const [activeTab, setActiveTab] = useState('summarizer');
  const [darkMode, setDarkMode] = useState(false);
  const [user, setUser] = useState(null);
  
  // Check auth state on load
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    const savedDark = localStorage.getItem('darkMode') === 'true';
    if (savedUser) setUser(JSON.parse(savedUser));
    setDarkMode(savedDark);
    if (savedDark) document.body.classList.add('dark');
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.body.classList.toggle('dark');
    localStorage.setItem('darkMode', (!darkMode).toString());
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/logout`, { method: 'POST' });
    } catch (e) {}
    setUser(null);
    localStorage.removeItem('user');
    setActiveTab('summarizer');
  };

  const tabs = [
    { id: 'summarizer', label: 'Summarizer', icon: FileText },
    { id: 'analyzer', label: 'Analyzer', icon: Search },
    { id: 'extractor', label: 'URL Extractor', icon: LinkIcon },
    { id: 'keywords', label: 'Keywords', icon: Key },
    { id: 'sentiment', label: 'Sentiment', icon: Smile },
  ];

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="logo">
          <FileText className="text-primary" />
          <span>Saransh AI</span>
        </div>
        <div className="nav-links">
          {tabs.map(tab => (
            <button 
              key={tab.id}
              className={activeTab === tab.id ? 'active' : ''}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
          <button 
            className={activeTab === 'profile' || activeTab === 'login' ? 'active' : ''}
            onClick={() => setActiveTab(user ? 'profile' : 'login')}
          >
            {user ? user.username : 'Login'}
          </button>
          <button className="icon-btn" onClick={toggleDarkMode}>
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </nav>

      <main className="main-content">
        {activeTab === 'summarizer' && <SummarizerTool user={user} />}
        {activeTab === 'analyzer' && <AnalyzerTool user={user} />}
        {activeTab === 'extractor' && <ExtractorTool user={user} />}
        {activeTab === 'keywords' && <KeywordsTool user={user} />}
        {activeTab === 'sentiment' && <SentimentTool user={user} />}
        {activeTab === 'login' && <Login onLogin={(u) => { setUser(u); localStorage.setItem('user', JSON.stringify(u)); setActiveTab('summarizer'); }} onSwitch={() => setActiveTab('register')} />}
        {activeTab === 'register' && <Register onSwitch={() => setActiveTab('login')} />}
        {activeTab === 'profile' && <Profile user={user} onLogout={handleLogout} />}
      </main>

      <footer className="footer">
        <p>© 2026 Saransh AI. Premium AI Workstation.</p>
      </footer>
    </div>
  );
}

// Reusable Hook for Tool API Calls
function useApiTool(endpoint) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const execute = async (payload) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Request failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { input, setInput, loading, result, error, execute };
}

function SummarizerTool() {
  const { input, setInput, loading, result, error, execute } = useApiTool('/summarize');
  return (
    <div className="tool-card">
      <h2><FileText size={28} /> Text Summarizer</h2>
      <div className="form-group">
        <label>Text or URL to Summarize</label>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Paste text here... (min 20 chars)" />
      </div>
      <button className="btn" onClick={() => execute({ text: input })} disabled={loading || input.length < 20}>
        {loading ? <Loader2 className="loader" /> : 'Summarize Content'}
      </button>
      {error && <div className="status-message error">{error}</div>}
      {result && <div className="output-box">{result.summary}</div>}
    </div>
  );
}

function AnalyzerTool() {
  const { input, setInput, loading, result, error, execute } = useApiTool('/analyze');
  return (
    <div className="tool-card">
      <h2><Search size={28} /> Text Analyzer</h2>
      <div className="form-group">
        <label>Text to Analyze</label>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Paste text here..." />
      </div>
      <button className="btn" onClick={() => execute({ text: input })} disabled={loading || !input}>
        {loading ? <Loader2 className="loader" /> : 'Analyze Content'}
      </button>
      {error && <div className="status-message error">{error}</div>}
      {result && result.analysis && (
        <div className="output-box">
          <p><strong>Word Count:</strong> {result.analysis.word_count}</p>
          <p><strong>Unique Words:</strong> {result.analysis.unique_words}</p>
          <p><strong>Sentences:</strong> {result.analysis.sentence_count}</p>
          <p><strong>Avg Word Length:</strong> {result.analysis.avg_word_length}</p>
          <p><strong>Readability:</strong> {result.analysis.readability}</p>
        </div>
      )}
    </div>
  );
}

function ExtractorTool() {
  const { input, setInput, loading, result, error, execute } = useApiTool('/extract');
  return (
    <div className="tool-card">
      <h2><LinkIcon size={28} /> URL Content Extractor</h2>
      <div className="form-group">
        <label>Website URL</label>
        <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="https://example.com" />
      </div>
      <button className="btn" onClick={() => execute({ url: input })} disabled={loading || !input}>
        {loading ? <Loader2 className="loader" /> : 'Extract Content'}
      </button>
      {error && <div className="status-message error">{error}</div>}
      {result && <div className="output-box">{result.content}</div>}
    </div>
  );
}

function KeywordsTool() {
  const { input, setInput, loading, result, error, execute } = useApiTool('/keywords');
  return (
    <div className="tool-card">
      <h2><Key size={28} /> Keyword Extractor</h2>
      <div className="form-group">
        <label>Text to Extract Keywords From</label>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Paste text here..." />
      </div>
      <button className="btn" onClick={() => execute({ text: input })} disabled={loading || !input}>
        {loading ? <Loader2 className="loader" /> : 'Extract Keywords'}
      </button>
      {error && <div className="status-message error">{error}</div>}
      {result && result.keywords && (
        <div className="output-box">
          <ul>{result.keywords.map((k, i) => <li key={i}>{k}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

function SentimentTool() {
  const { input, setInput, loading, result, error, execute } = useApiTool('/sentiment');
  return (
    <div className="tool-card">
      <h2><Smile size={28} /> Sentiment Analyzer</h2>
      <div className="form-group">
        <label>Text to Analyze</label>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Paste text here..." />
      </div>
      <button className="btn" onClick={() => execute({ text: input })} disabled={loading || !input}>
        {loading ? <Loader2 className="loader" /> : 'Analyze Sentiment'}
      </button>
      {error && <div className="status-message error">{error}</div>}
      {result && result.sentiment && (
        <div className="output-box">
          <p><strong>Overall Sentiment:</strong> {result.sentiment.sentiment}</p>
          <p><strong>Positive Score:</strong> {result.sentiment.positive}</p>
          <p><strong>Negative Score:</strong> {result.sentiment.negative}</p>
          <p><strong>Neutral Score:</strong> {result.sentiment.neutral}</p>
        </div>
      )}
    </div>
  );
}

function Login({ onLogin, onSwitch }) {
  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identifier, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      onLogin(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tool-card" style={{ maxWidth: '400px' }}>
      <h2><User size={28} /> Welcome Back</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Username or Email</label>
          <input type="text" value={identifier} onChange={e => setIdentifier(e.target.value)} required />
        </div>
        <div className="form-group">
          <label>Password</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} required />
        </div>
        <button type="submit" className="btn" disabled={loading}>
          {loading ? <Loader2 className="loader" /> : 'Sign In'}
        </button>
      </form>
      {error && <div className="status-message error">{error}</div>}
      <p style={{ marginTop: '1rem', textAlign: 'center' }}>
        New here? <button onClick={onSwitch} style={{ background:'none', border:'none', color:'var(--primary)', cursor:'pointer', fontWeight:600 }}>Create an account</button>
      </p>
    </div>
  );
}

function Register({ onSwitch }) {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      setSuccess(true);
      setTimeout(onSwitch, 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="tool-card" style={{ maxWidth: '400px' }}>
      <h2><User size={28} /> Create Account</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Username</label>
          <input type="text" value={username} onChange={e => setUsername(e.target.value)} required minLength={3} />
        </div>
        <div className="form-group">
          <label>Email</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} required />
        </div>
        <div className="form-group">
          <label>Password</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={5} />
        </div>
        <button type="submit" className="btn" disabled={loading}>
          {loading ? <Loader2 className="loader" /> : 'Sign Up'}
        </button>
      </form>
      {error && <div className="status-message error">{error}</div>}
      {success && <div className="status-message success">Registration successful! Redirecting...</div>}
      <p style={{ marginTop: '1rem', textAlign: 'center' }}>
        Already have an account? <button onClick={onSwitch} style={{ background:'none', border:'none', color:'var(--primary)', cursor:'pointer', fontWeight:600 }}>Sign In</button>
      </p>
    </div>
  );
}

function Profile({ user, onLogout }) {
  const [pref, setPref] = useState(user.preferences || '150');
  const [msg, setMsg] = useState('');

  const savePref = async () => {
    try {
      await fetch(`${API_BASE}/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ summary_length: pref })
      });
      setMsg('Preferences saved successfully!');
      setTimeout(() => setMsg(''), 3000);
    } catch (e) {
      setMsg('Error saving preferences.');
    }
  };

  return (
    <div className="tool-card" style={{ maxWidth: '500px' }}>
      <h2><User size={28} /> Your Profile</h2>
      <div style={{ marginBottom: '2rem' }}>
        <p><strong>Username:</strong> {user.username}</p>
        <p><strong>Member Since:</strong> {user.joinDate}</p>
      </div>
      
      <div className="form-group">
        <label>Summary Length Preference</label>
        <select value={pref} onChange={(e) => setPref(e.target.value)}>
          <option value="100">Short (100 words)</option>
          <option value="150">Medium (150 words)</option>
          <option value="200">Long (200 words)</option>
        </select>
      </div>
      
      <button className="btn" onClick={savePref} style={{ marginBottom: '1rem' }}>
        Save Preferences
      </button>
      
      {msg && <div className="status-message success">{msg}</div>}
      
      <button className="btn" onClick={onLogout} style={{ background: 'var(--error)' }}>
        <LogOut size={20} /> Sign Out
      </button>
    </div>
  );
}

export default App;
