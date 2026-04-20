# 🤖 Ollama-Agent

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
git clone https://github.com/declan2010/ollama-agent.git
cd ollama-agent

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
ollama-agent/
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

## Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully supported | Native support, runs as-is |
| **macOS** | ✅ Fully supported | Same as Linux, no changes needed |
| **Windows** | ⚠️ Requires WSL2 | See below |

### Running on Windows

Ollama-Agent uses Linux/Unix shell commands (`ls`, `cat`, `grep`, `bash`, etc.) and Unix-style file paths (`/home/user/`). It does **not** run natively on Windows Command Prompt or PowerShell.

To run on Windows, use **one of these options**:

1. **WSL2 (Recommended)** — Run inside Windows Subsystem for Linux:
   ```bash
   # Install WSL2 if you haven't
   wsl --install
   # Then inside WSL2:
   sudo apt install python3 python3-pip
   pip install flask flask-cors ddgs
   python3 ollama_chat.py
   ```
   Make sure Ollama is installed on Windows and accessible from WSL2 (usually at `http://localhost:11434`).

2. **Docker** — Run in a Linux container:
   ```bash
   docker run -it -p 5000:5000 -w /app -v $(pwd):/app python:3.12-slim \
     bash -c "pip install flask flask-cors ddgs && python ollama_chat.py"
   ```

3. **Git Bash** — May work for basic usage but some commands may fail. Not officially supported.

> **Note:** Native Windows support (PowerShell/CMD) is not currently planned. If you need it, contributions are welcome.

## ☕ Support This Project

Ollama-Agent is free and open source. If you find it useful, consider donating to help cover development and server costs. Your support keeps this project alive and improving!

[![PayPal](https://img.shields.io/badge/PayPal-donate-blue)](https://www.paypal.com/donate/?hosted_button_id=P3TK84JYLNUU6)

## Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. IN NO EVENT SHALL THE DEVELOPER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE. BY INSTALLING AND/OR USING OLLAMAAGENT, YOU ACKNOWLEDGE AND ACCEPT FULL RESPONSIBILITY FOR ANY DAMAGE TO PERSONAL OR CORPORATE DATA, SYSTEMS, OR EQUIPMENT THAT MAY RESULT FROM ITS OPERATION. THE AI AGENT HAS THE ABILITY TO EXECUTE SYSTEM COMMANDS AND MODIFY FILES — USE ENTIRELY AT YOUR OWN RISK.

## License

MIT License - see [LICENSE](LICENSE) file.