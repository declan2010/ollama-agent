# 🤖 OllamaAgent

**by Declan2010**

A web-based AI agent for Ollama with web search, local command execution, and file management capabilities.

## Features

- **Web Interface**: Modern chat UI with real-time streaming responses
- **Web Search**: Search the internet via DuckDuckGo
- **Local Commands**: Execute read and write commands on the host system
- **File Management**: Create, edit, and write files directly from the agent
- **Session Management**: Chat history persisted locally per conversation
- **Model Selection**: Choose from available Ollama models (local & cloud)
- **Fallback Models**: Automatic fallback if primary model fails
- **Markdown Rendering**: Full markdown support with syntax highlighting
- **Dark/Light Theme**: Toggle between themes

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) running locally (default: `http://localhost:11434`)
- Ollama models installed (e.g., `ollama pull llama3`)

## Installation

```bash
# Clone the repository
git clone https://github.com/declan2010/ollamaAgent.git
cd ollamaAgent

# Install dependencies
pip install flask flask-cors ddgs

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

## Available Tools

The agent can use these tools when available:

| Tool | Description |
|------|-------------|
| `local_command` | Execute system commands (read-only auto, write with notification) |
| `web_search` | Search the internet via DuckDuckGo |
| `fetch_article` | Fetch and read web page content |

## Project Structure

```
ollamaAgent/
├── ollama_chat.py   # Flask backend with Ollama integration
├── templates/
│   └── index.html   # Frontend chat interface
├── README.md         # This file
└── LICENSE           # MIT License
```

## Security Notes

- Sessions are stored locally on disk
- Read commands execute automatically (ls, cat, grep, etc.)
- Write commands (touch, cp, mv, etc.) auto-execute with notification
- No authentication by default (add reverse proxy for production)

## ☕ Support This Project

OllamaAgent is free and open source. If you find it useful, consider donating to help cover development and server costs. Your support keeps this project alive and improving!

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-orange)](https://buymeacoffee.com/declan2010)

## Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. IN NO EVENT SHALL THE DEVELOPER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE. BY INSTALLING AND/OR USING OLLAMAAGENT, YOU ACKNOWLEDGE AND ACCEPT FULL RESPONSIBILITY FOR ANY DAMAGE TO PERSONAL OR CORPORATE DATA, SYSTEMS, OR EQUIPMENT THAT MAY RESULT FROM ITS OPERATION. THE AI AGENT HAS THE ABILITY TO EXECUTE SYSTEM COMMANDS AND MODIFY FILES — USE ENTIRELY AT YOUR OWN RISK.

## License

MIT License - see [LICENSE](LICENSE) file.