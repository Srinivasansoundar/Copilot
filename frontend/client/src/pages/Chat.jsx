
import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function Chat() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dataSummary, setDataSummary] = useState(null);
  const [generateDiagram, setGenerateDiagram] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Add welcome message on mount
    setMessages([{
      type: 'assistant',
      content: 'Hello! I can help you analyze your CSV data. What would you like to know? You can ask for diagrams by including words like "chart," "plot," "graph," or "visualize" in your question.'
    }]);
    fetchDataSummary();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchDataSummary = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/api/data/summary', {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch data summary');
      }

      const data = await response.json();
      setDataSummary(data.summary);
    } catch (error) {
      console.error('Error fetching data summary:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = { type: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/api/chat/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          query: query,
          generate_diagram: generateDiagram
        })
      });

      if (!response.ok) throw new Error('Failed to process query');
      const data = await response.json();

      const assistantMessage = {
        type: 'assistant',
        content: data.response.answer,
        diagram: null,
        visualization: null
      };

      if (data.visualization) {
        if (data.visualization.status === 'success') {
          assistantMessage.diagram = data.visualization.diagram_path;
          assistantMessage.visualization = {
            type: data.visualization.type,
            title: data.visualization.title,
            x_axis: data.visualization.x_axis,
            y_axis: data.visualization.y_axis
          };
        } else if (data.visualization.status === 'error') {
          assistantMessage.content += `\n\nNote: There was an issue generating the visualization: ${data.visualization.error_message}`;
          if (data.visualization.fallback_path) {
            assistantMessage.diagram = data.visualization.fallback_path;
          }
        }
      }

      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      console.error('Error processing query:', err);
      setMessages(prev => [...prev, {
        type: 'system',
        content: 'Sorry, there was an error processing your query. Please try again.'
      }]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  const toggleDiagramGeneration = () => {
    setGenerateDiagram(!generateDiagram);
  };

  return (
    <div className="app-container">
      <header>
        <h1>Shipper Copilot</h1>
        <div className="header-controls">
          <label className="diagram-toggle">
            <input 
              type="checkbox" 
              checked={generateDiagram} 
              onChange={toggleDiagramGeneration}
            />
            Auto-generate diagrams
          </label>
        </div>
      </header>
      
      <div className="main-content">
        <div className="chat-container">
          <div className="messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">
                  {String(msg.content).split('\n').map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                  {msg.diagram && (
                    <div className="diagram-container">
                      <div className="diagram-title">
                        {msg.visualization?.title || 'Generated Visualization'}
                      </div>
                      <img 
                        src={`http://localhost:5000${msg.diagram}`} 
                        alt="Data visualization"
                      />
                      {msg.visualization && (
                        <div className="diagram-info">
                          <p>Type: {msg.visualization.type}</p>
                          {msg.visualization.x_axis && <p>X-axis: {msg.visualization.x_axis}</p>}
                          {msg.visualization.y_axis && <p>Y-axis: {msg.visualization.y_axis}</p>}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <div className="message-content loading">
                  <div className="loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="query-form">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your CSV data..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !query.trim()}>
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Chat;




