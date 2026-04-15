# Ollama WebChat

A web-based chat interface for Ollama with local system commands and web search capabilities.

## Features

- **Web Interface**: Modern chat UI with real-time responses
- **Local Commands**: Execute read-only system commands (ls, df, free, etc.)
- **Web Search**: Search the internet via DuckDuckGo
- **Session Management**: Chat history persisted locally
- **Model Selection**: Choose from available Ollama models
- **Fallback Models**: Automatic fallback if primary model fails

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally (default: `http://localhost:11434`)
- Ollama models installed (e.g., `ollama pull llama3`)

## Installation

```bash
# Clone the repository
git clone https://github.com/declan2010/ollama-webchat.git
cd ollama-webchat

# Install dependencies
pip install flask flask-cors duckduckgo-search

# Run the server
python ollama_chat.py
```

Open your browser at: **http://localhost:5000**

## Configuration

### Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `SESSION_DIR` | `./sessions` | Directory for chat sessions |
| `PORT` | `5000` | Server port |

### Example

```bash
export OLLAMA_HOST=http://localhost:11434
export PORT=5000
python ollama_chat.py
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [Flask](https://flask.palletsprojects.com/) | >=2.0 | Web framework |
| [Flask-CORS](https://flask-cors.readthedocs.io/) | >=3.0 | CORS support |
| [ddgs](https://pypi.org/project/ddgs/) | >=9.0 | DuckDuckGo search (renamed from duckduckgo-search) |

Install all dependencies:
```bash
pip install flask flask-cors ddgs
```

## Project Structure

```
ollama-webchat/
├── ollama_chat.py   # Flask backend with Ollama integration
├── templates/
│   └── index.html   # Frontend chat interface
├── README.md         # This file
├── LICENSE           # MIT License
└── CONTRIBUTING.md   # Contribution guidelines
```

## API Endpoints

### POST /api/chat
Send a message and receive a response.

**Request:**
```json
{
  "message": "Hello, what is the weather?",
  "model": "llama3",
  "fallback_model": "mistral"
}
```

**Response:**
```json
{
  "response": "The weather is sunny...",
  "session_id": "abc12345"
}
```

### GET /api/sessions
List all saved chat sessions.

### GET /api/session/<session_id>
Load a specific chat session.

### DELETE /api/session/<session_id>
Delete a chat session.

### GET /api/models
List available Ollama models.

## Available Tools

The assistant can use these tools when available:

| Tool | Description |
|------|-------------|
| `web_search` | Search the internet via DuckDuckGo for current information, news, and facts |

## Security Notes

- Sessions are stored locally on disk
- No authentication by default (add reverse proxy for production)

## License

MIT License - see [LICENSE](LICENSE) file.