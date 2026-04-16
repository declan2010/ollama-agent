"""
Ollama WebChat - Web chat for local Ollama models
Allows chatting with local models, selecting models, and saving sessions.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'ollama-chat-secret-key-2024'
SESSIONS_DIR = 'sessions'

# Create sessions directory if it does not exist
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

# Allowed local commands (read-only)
ALLOWED_COMMANDS = {
    'ls': ['ls', '-la', '-l', '-a', '-R'],
    'pwd': ['pwd'],
    'whoami': ['whoami'],
    'date': ['date'],
    'df': ['df', 'df -h', 'df -T'],
    'free': ['free', 'free -h', 'free -m'],
    'uname': ['uname', 'uname -a'],
    'ps': ['ps', 'ps aux'],
    'top': ['top', 'top -n 1'],
    'cat': ['cat '],  # with file
    'head': ['head ', 'head -n '],
    'tail': ['tail ', 'tail -n '],
    'wc': ['wc ', 'wc -l '],
    'du': ['du', 'du -sh'],
    'tree': ['tree', 'tree -L '],
    'find': ['find . -'],
    'hostname': ['hostname'],
    'uptime': ['uptime'],
    'netstat': ['netstat', 'netstat -tuln'],
    'ifconfig': ['ifconfig', 'ip addr'],
    'curl': ['curl '],  # for connectivity check
    'ping': ['ping -c 4 '],
}

# Common translations for local commands
TRANSLATIONS = {
    # Spanish
    'es': {
        'archivos': 'files', 'directorio': 'directory', 'carpeta': 'folder',
        'actual': 'current', 'mostrar': 'show', 'ver': 'view', 'listar': 'list',
        'disco': 'disk', 'espacio': 'space', 'almacenamiento': 'storage',
        'memoria': 'memory', 'ram': 'ram', 'usuario': 'user', 'nombre': 'name',
        'sistema': 'system', 'computadora': 'computer', 'equipo': 'machine',
        'info': 'info', 'informacion': 'information', 'cual': 'which',
        'donde': 'where', 'hay': 'there', 'existen': 'exists',
        'cuanto': 'how much', 'estado': 'status', 'uso': 'usage',
        'procesos': 'processes', 'red': 'network', 'conexion': 'connection',
        'direccion ip': 'ip address', 'nombre del equipo': 'hostname',
        'tiempo activo': 'uptime', 'desde cuando': 'how long',
    },
    # Portuguese
    'pt': {
        'arquivos': 'files', 'diretorio': 'directory', 'pasta': 'folder',
        'atual': 'current', 'mostrar': 'show', 'ver': 'view', 'listar': 'list',
        'disco': 'disk', 'espaco': 'space', 'memoria': 'memory',
        'usuario': 'user', 'sistema': 'system', 'computador': 'computer',
        'info': 'info', 'qual': 'which', 'onde': 'where', 'tem': 'there',
        'quanto': 'how much', 'rede': 'network', 'conexao': 'connection',
    },
    # French
    'fr': {
        'fichiers': 'files', 'repertoire': 'directory', 'dossier': 'folder',
        'actuel': 'current', 'afficher': 'show', 'voir': 'view', 'lister': 'list',
        'disque': 'disk', 'espace': 'space', 'stockage': 'storage',
        'memoire': 'memory', 'ram': 'ram', 'utilisateur': 'user',
        'systeme': 'system', 'ordinateur': 'computer', 'machine': 'machine',
        'info': 'info', 'information': 'information', 'quel': 'which',
        'ou': 'where', 'combien': 'how much', 'reseau': 'network',
    },
    # German
    'de': {
        'dateien': 'files', 'verzeichnis': 'directory', 'ordner': 'folder',
        'aktuell': 'current', 'anzeigen': 'show', 'zeigen': 'show',
        'auflisten': 'list', 'festplatte': 'disk', 'speicher': 'space',
        'speicherplatz': 'storage', 'gedaechtnis': 'memory', 'ram': 'ram',
        'benutzer': 'user', 'system': 'system', 'computer': 'computer',
        'info': 'info', 'information': 'information', 'welcher': 'which',
        'wie viel': 'how much', 'netzwerk': 'network', 'verbindung': 'connection',
    },
    # Italian
    'it': {
        'file': 'files', 'directory': 'directory', 'cartella': 'folder',
        'attuale': 'current', 'mostrare': 'show', 'vedere': 'view', 'elenco': 'list',
        'disco': 'disk', 'spazio': 'space', 'memoria': 'memory', 'ram': 'ram',
        'utente': 'user', 'sistema': 'system', 'computer': 'computer',
        'info': 'info', 'informazioni': 'information', 'quale': 'which',
        'rete': 'network', 'connessione': 'connection',
    },
    # Catalan
    'ca': {
        'arxius': 'files', 'directori': 'directory', 'carpeta': 'folder',
        'actual': 'current', 'mostrar': 'show', 'veure': 'view', 'llistar': 'list',
        'disc': 'disk', 'espai': 'space', 'emmagatzematge': 'storage',
        'memoria': 'memory', 'ram': 'ram', 'usuari': 'user',
        'sistema': 'system', 'ordinador': 'computer', 'maquina': 'machine',
    },
}

def translate_message(message):
    """Translate message to English using simple dictionaries"""
    msg_lower = message.lower()

    # Collect all translations found
    all_translations = {}

    # Combine all dictionaries
    for lang, dict_trans in TRANSLATIONS.items():
        all_translations.update(dict_trans)

    # Sort by length (longest first) to avoid partial replacements
    sorted_words = sorted(all_translations.keys(), key=len, reverse=True)

    # Apply translations
    translated = msg_lower
    for native in sorted_words:
        english = all_translations[native]
        # Replace with word boundaries
        import re
        pattern = r'\b' + re.escape(native) + r'\b'
        translated = re.sub(pattern, english, translated)

    return translated

# Patterns that trigger local commands (English and translated)
LOCAL_CMD_PATTERNS = [
    # Files/Directory - keywords
    (r'\b(list|show|get|see|view|display)\b.*\b(files|contents)\b', 'ls -la'),
    (r'\b(files|contents|archivos|contenido)\b.*\b(directory|folder|here|there|actual|current)\b', 'ls -la'),
    (r'\b(all files|list files|show files|ver archivos|mostrar archivos|afficher fichiers|voir fichiers|anzeigen|dateien|mostrare file|llistar arxius)\b', 'ls -la'),
    (r'\b(what.*there|which.*files|cuales.*archivos|quels.*fichiers|welche.*dateien|quali.*file)\b', 'ls -la'),

    # Disk/Space
    (r'\b(disk|space|storage|disco|espacio|almacenamiento|disque|espace|stockage|festplatte|speicher|speicherplatz|disc|espai|emmagatzematge)\b', 'df -h'),
    (r'\b(how much|cuanto|cuanta|combien|wie viel|quanto|quanta)\b.*\b(free|available|libre|free)\b', 'df -h'),

    # Memory
    (r'\b(memory|ram|memoria|gedaechtnis|memoire)\b', 'free -h'),
    (r'\b(how much|cuanto|cuanta|combien|wie viel|quanto|quanta).*\b(available|free|used)\b', 'free -h'),

    # User
    (r'\b(who am i|username|user name|quien soy|qui suis-je|wer bin ich|chi sono)\b', 'whoami'),

    # Current directory
    (r'\b(where|donde|ou|wo|dove).*\b(am i|estoy|suis|bin|sono)\b', 'pwd'),
    (r'\b(current|present|actual|actuel|aktuell|attuale).*\b(location|place|directory|folder)\b', 'pwd'),
    (r'\b(pwd|cwd|directorio actual|carpeta actual|repertoire actuel|aktuelles verzeichnis|dove sono)\b', 'pwd'),

    # System info
    (r'\b(system|computer|machine|sistema|computadora|equipo|systeme|ordinateur|maschine|computer|maquina)\b', 'uname -a'),
    (r'\b(info|information|details|detalles|informations|details|informazioni|dettagli)\b', 'uname -a'),

    # Uptime
    (r'\b(uptime|tiempo activo|desde cuando|depuis combien|wie lange|da quanto)\b', 'uptime'),

    # Network/IP
    (r'\b(ip|network|red|reseau|netzwerk|rete)\b', 'ip addr'),

    # Hostname
    (r'\b(hostname|computer name|nombre equipo|nom ordinateur|computername|nome computer)\b', 'hostname'),
]

# Dangerous commands we do NOT allow
BLOCKED_PATTERNS = [
    r'\b(rm|del|delete|rm -rf|mkfs|dd|wipe|destroy|supprimer|entfernen|eliminare)\b',
    r'\b(sudo|su|chmod 777|chown)\b',
    r'\b(wget|curl.*\|.*sh|bash.*http)\b',
    r'\b(mkdir /|touch /|echo > /)\b',
    r'\b(ssh|scp|rsync)\b',
    r'\b(sql|nmap|nikto|hydra)\b',
]

def is_dangerous(cmd):
    """Check if the command is dangerous"""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False

def detect_local_command(message):
    """Detect if message asks to execute a local command (multilingual support)"""
    # First translate to English
    translated = translate_message(message)

    # Check patterns in translated message
    for pattern, cmd in LOCAL_CMD_PATTERNS:
        if re.search(pattern, translated):
            return cmd

    # Check if it's exactly an allowed command
    for base_cmd in ALLOWED_COMMANDS.keys():
        if translated.startswith(base_cmd + ' ') or translated == base_cmd:
            if is_dangerous(translated):
                return None
            return translated

    return None

def execute_local_command(cmd):
    """Execute a local command (read-only only)"""
    try:
        # Normalize command
        cmd = cmd.strip()
        if is_dangerous(cmd):
            return "[Security] This command is not allowed."

        # Execute with timeout
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.expanduser('~')
        )

        output = result.stdout.strip() or result.stderr.strip() or "Command executed successfully (no output)"
        return output[:5000]  # Limit response to 5000 chars

    except subprocess.TimeoutExpired:
        return "[Timeout] Command took too long (>10s)"
    except Exception as e:
        return f"[Error] {str(e)}"

def get_ollama_models():
    """Get list of available models"""
    try:
        import urllib.request
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        return []

def send_to_ollama(model, messages, tools=None):
    """Send message to Ollama and return response, with tools support"""
    try:
        import urllib.request

        payload = {
            'model': model,
            'messages': messages,
            'stream': False
        }

        # Add tools if provided
        if tools:
            payload['tools'] = tools

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:11434/api/chat',
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        # Make request with long timeout for tool calls
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read())
            return result
    except Exception as e:
        return {'error': str(e)}

def process_ollama_response(model, messages, tools=None):
    """Process Ollama response, executing tools if necessary"""
    max_iterations = 2  # Reduced to avoid infinite loops

    for i in range(max_iterations):
        response = send_to_ollama(model, messages, tools)

        if 'error' in response:
            return f"Error: {response['error']}"

        # Get response message
        assistant_msg = response.get('message', {})
        content = assistant_msg.get('content', '')
        tool_calls = assistant_msg.get('tool_calls', [])

        # If no tool calls, return content directly
        if not tool_calls:
            return content

        # Process each tool call
        tool_results = []
        for tool_call in tool_calls:
            func_name = tool_call.get('function', {}).get('name', '')
            func_args = tool_call.get('function', {}).get('arguments', {})
            tool_id = tool_call.get('id', f'tool_{i}')

            print(f"  [TOOL CALL #{i+1}] {func_name} with args: {func_args}")

            # Execute function by name
            if func_name == 'local_command':
                cmd = func_args.get('command', '')
                result = execute_local_command(cmd)
                tool_results.append({
                    'role': 'tool',
                    'content': result,
                    'tool_call_id': tool_id
                })
            elif func_name == 'web_search':
                query = func_args.get('query', '')
                results = web_search(query)
                if results and isinstance(results, list) and 'error' in results[0]:
                    result = f"Search error: {results[0]['error']}"
                else:
                    result = "Search results:\n\n"
                    for idx, r in enumerate(results[:5], 1):
                        result += f"{idx}. {r['title']}\n   URL: {r['url']}\n"
                        if r.get('full_content'):
                            result += f"   Content: {r['full_content']}\n\n"
                        else:
                            result += f"   {r['snippet']}\n\n"
                tool_results.append({
                    'role': 'tool',
                    'content': result,
                    'tool_call_id': tool_id
                })
            elif func_name == 'fetch_article':
                url = func_args.get('url', '')
                article = fetch_article(url)
                if 'content' in article:
                    result = f"Article from {url}:\n\n{article['content']}"
                else:
                    result = f"Could not fetch article from {url}: {article.get('error', 'Unknown error')}"
                tool_results.append({
                    'role': 'tool',
                    'content': result,
                    'tool_call_id': tool_id
                })
            else:
                tool_results.append({
                    'role': 'tool',
                    'content': f"Unknown tool: {func_name}",
                    'tool_call_id': tool_id
                })

        # Add assistant response
        messages.append({
            'role': 'assistant',
            'content': content,
            'tool_calls': tool_calls  # Include tool_calls in assistant message
        })

        # Add tool results
        for tr in tool_results:
            messages.append(tr)

        # If there was only one tool call and we processed it, return result directly
        if len(tool_calls) == 1 and content.strip() == '':
            return f"Command executed:\n\n{tool_results[0]['content']}"

    # If we reached max, return last tool result
    if tool_results:
        return f"Command executed:\n\n{tool_results[0]['content']}"

    return "I couldn't complete the request. Please try a simpler question."

# Ollama tools definition
OLLAMA_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'local_command',
            'description': 'Execute a read-only local system command. Only use for: listing files, checking disk space, viewing memory, system info, network status, user info. NEVER use for: rm, del, sudo, chmod, dd, mkfs, or any write/modify/destructive operations.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The command to execute. Examples: "ls -la", "df -h", "free -h", "uname -a", "pwd", "whoami", "uptime", "hostname"'
                    }
                },
                'required': ['command']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the internet for information. Use this when you need to find current information, news, facts, or answers that require up-to-date data from the web. Returns title, URL, and snippet for each result. IMPORTANT: After searching, always use fetch_article on the most relevant URLs to get full content before answering.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query to find information on the internet. Be specific and include key terms.'
                    }
                },
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'fetch_article',
            'description': 'Fetch and extract the full text content from a web article URL. Use this after web_search to get detailed content from the most relevant articles before answering the user. Returns the article text content.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'The URL of the article to fetch and extract content from'
                    }
                },
                'required': ['url']
            }
        }
    }
]


def web_search(query):
    """Search the internet using DuckDuckGo (ddgs) and fetch full article content"""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=5):
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')[:300]
                })
            
            # Auto-fetch full content from top 3 results
            for i in range(min(3, len(results))):
                article = fetch_article(results[i]['url'])
                if 'content' in article and article['content']:
                    results[i]['full_content'] = article['content'][:2000]
            
            return results
    except Exception as e:
        return [{'error': str(e)}]


def fetch_article(url):
    """Fetch and extract text content from a web article URL"""
    try:
        import urllib.request
        import html as html_mod
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
        resp = urllib.request.urlopen(req, timeout=10)
        html_content = resp.read().decode("utf-8", errors="ignore")
        # Remove scripts, styles, nav, header, footer
        for t in ["script", "style", "nav", "header", "footer", "aside", "noscript"]:
            html_content = re.sub(f"<{t}[^>]*>.*?</{t}>", "", html_content, flags=re.DOTALL|re.IGNORECASE)
        # Extract paragraphs
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_content, re.DOTALL|re.IGNORECASE)
        lines = []
        for p in paragraphs:
            clean = re.sub(r"<[^>]+>", "", p).strip()
            clean = html_mod.unescape(clean)
            if len(clean) > 50:
                lines.append(clean)
        text = "\n".join(lines)
        if not text:
            # Fallback: extract all text from body
            body = re.search(r"<body[^>]*>(.*?)</body>", html_content, re.DOTALL|re.IGNORECASE)
            if body:
                text = html_mod.unescape(re.sub(r"<[^>]+>", " ", body.group(1)))
                text = re.sub(r"\s+", " ", text).strip()[:3000]
        else:
            text = text[:3000]
        if text:
            return {'url': url, 'content': text}
        return {'url': url, 'error': 'Could not extract content'}
    except Exception as e:
        return {'url': url, 'error': str(e)}


def save_session(session_id, data):
    """Save session to JSON file"""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(session_id):
    """Load session from JSON file"""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def list_sessions():
    """List all saved sessions"""
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith('.json'):
            session_id = filename[:-5]
            data = load_session(session_id)
            if data:
                sessions.append({
                    'id': session_id,
                    'model': data.get('model', 'unknown'),
                    'title': data.get('title', 'Untitled'),
                    'created': data.get('created', ''),
                    'messages_count': len(data.get('messages', []))
                })
    # Sort by most recent date
    sessions.sort(key=lambda x: x.get('created', ''), reverse=True)
    return sessions

@app.route('/')
def index():
    """Main page"""
    models = get_ollama_models()
    sessions = list_sessions()

    # Create new session if none exists
    if 'chat_id' not in session:
        import uuid
        session['chat_id'] = str(uuid.uuid4())[:8]
        session['model'] = models[0] if models else 'llama3'

    return render_template('index.html',
                         models=models,
                         sessions=sessions,
                         current_model=session.get('model', ''))

@app.route('/api/models')
def api_models():
    """API to get models"""
    return jsonify(get_ollama_models())

@app.route('/api/model-info')
def api_model_info():
    """API to get model info including context"""
    model_name = request.args.get('model', '')
    try:
        import urllib.request
        req = urllib.request.Request(
            f'http://localhost:11434/api/show',
            data=json.dumps({'name': model_name}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())

            # Search context length in several places
            num_ctx = data.get('num_ctx') or data.get('context_length')

            # Search in model_info (some models put it there)
            model_info = data.get('model_info', {})
            if not num_ctx:
                for key in model_info:
                    if 'context_length' in key.lower():
                        num_ctx = model_info[key]
                        break

            return jsonify({
                'model': model_name,
                'context_length': num_ctx or 4096,
                'num_ctx': num_ctx or 4096,
                'model_info': model_info,
                'details': data.get('details', {})
            })
    except Exception as e:
        return jsonify({'error': str(e), 'model': model_name})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API to send message and receive response"""
    data = request.json
    user_message = data.get('message', '').strip()
    model = data.get('model', session.get('model', 'llama3'))
    fallback_model = data.get('fallback_model', '')

    if not user_message:
        return jsonify({'error': 'Empty message'})

    # Create session if not exists
    if 'chat_id' not in session:
        import uuid
        session['chat_id'] = str(uuid.uuid4())[:8]

    # Update model in session
    session['model'] = model
    if fallback_model:
        session['fallback_model'] = fallback_model

    # Load or create session
    session_data = load_session(session['chat_id']) or {
        'model': model,
        'fallback_model': fallback_model,
        'title': user_message[:50] + ('...' if len(user_message) > 50 else ''),
        'created': datetime.now().isoformat(),
        'messages': []
    }

    # Add user message
    session_data['messages'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })

    # Check if it's a local command (simple patterns that always work)
    local_cmd = detect_local_command(user_message)

    if local_cmd:
        # Execute local command directly (simple patterns)
        result = execute_local_command(local_cmd)
        response_text = f"[LOCAL COMMAND: {local_cmd}]\n\n{result}"
    else:
        # Use process_ollama_response which handles native Ollama tools
        # Try primary model, if fails use fallback
        response_text = process_ollama_response(model, session_data['messages'], OLLAMA_TOOLS)

        # If error and there's a fallback, try with fallback
        if response_text.startswith('[ERROR]') and fallback_model and fallback_model != model:
            print(f"Primary model '{model}' failed, trying fallback '{fallback_model}'")
            response_text = process_ollama_response(fallback_model, session_data['messages'], OLLAMA_TOOLS)
            if not response_text.startswith('[ERROR]'):
                response_text = f"[Fallback: {fallback_model}]\n\n{response_text}"

    # Add response
    session_data['messages'].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': datetime.now().isoformat()
    })

    # Save session
    save_session(session['chat_id'], session_data)

    return jsonify({
        'response': response_text,
        'session_id': session['chat_id']
    })

@app.route('/api/sessions')
def api_sessions():
    """API to list all sessions"""
    return jsonify(list_sessions())

@app.route('/api/session/<session_id>')
def api_session_get(session_id):
    """API to get a specific session"""
    data = load_session(session_id)
    if data:
        return jsonify(data)
    return jsonify({'error': 'Session not found'})

@app.route('/api/session/delete', methods=['POST'])
def api_session_delete():
    """API to delete a session"""
    data = request.json
    session_id = data.get('session_id', '')

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True})

    return jsonify({'error': 'Session not found'})

@app.route('/api/session/rename', methods=['POST'])
def api_session_rename():
    """API to rename a session"""
    data = request.json
    session_id = data.get('session_id', '')
    new_title = data.get('title', '')

    if not new_title:
        return jsonify({'error': 'Title cannot be empty'})

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        session_data['title'] = new_title
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        return jsonify({'success': True})

    return jsonify({'error': 'Session not found'})

@app.route('/api/session/new', methods=['POST'])
def api_session_new():
    """API to create a new session"""
    import uuid
    session['chat_id'] = str(uuid.uuid4())[:8]

    # Clear session data
    session_data = {
        'model': session.get('model', 'llama3'),
        'title': 'New conversation',
        'created': datetime.now().isoformat(),
        'messages': []
    }
    save_session(session['chat_id'], session_data)

    return jsonify({
        'session_id': session['chat_id'],
        'title': session_data['title']
    })

if __name__ == '__main__':
    # Create sessions directory if not exists
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)

    print("=" * 50)
    print("Ollama WebChat")
    print("=" * 50)
    print("Open: http://localhost:5000")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)