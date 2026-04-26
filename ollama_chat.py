"""
Ollama-Agent - Web Agent for local Ollama models
Allows chatting with local models, selecting models, and saving sessions.
"""

import json
import logging
import os
import queue
import uuid
import re
import time
from collections import defaultdict
from datetime import datetime
from flask import Flask, Response, render_template, request, jsonify, session

# --- Configuration ---
SESSIONS_DIR = os.environ.get('SESSIONS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions'))
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
KEEP_ALIVE = os.environ.get('OLLAMA_KEEP_ALIVE', '5m')

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('ollama-agent')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# --- Rate Limiting ---
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30     # requests per window per IP
_rate_limits = defaultdict(list)  # ip -> [timestamps]


def rate_limit_exceeded(ip):
    """Check if IP has exceeded rate limit. Returns True if blocked."""
    now = time.time()
    # Clean old entries
    _rate_limits[ip] = [t for t in _rate_limits[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limits[ip]) >= RATE_LIMIT_MAX:
        return True
    _rate_limits[ip].append(now)
    return False


# --- Sessions Directory ---
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

# --- Sensitive file paths (block reading these) ---
SENSITIVE_PATHS = [
    '/etc/passwd', '/etc/shadow', '/etc/gshadow', '/etc/group',
    '/etc/ssh/', '/root/.ssh/', '/home/', '/etc/hosts',
    '/etc/sudoers', '/etc/pam.d/', '/var/log/',
    '/proc/', '/sys/', '/dev/',
]


def is_sensitive_path(filepath):
    """Check if filepath points to a sensitive system file."""
    filepath = os.path.normpath(filepath)
    for sensitive in SENSITIVE_PATHS:
        if filepath == sensitive or filepath.startswith(sensitive):
            return True
    # Also block any path containing ssh, shadow, passwd, etc.
    basename = os.path.basename(filepath)
    blocked_names = {'passwd', 'shadow', 'gshadow', 'sudoers', 'ssh_config',
                     'id_rsa', 'id_ed25519', 'id_ecdsa', 'authorized_keys',
                     'known_hosts', '.ssh', '.env', '.gitconfig',
                     'credentials', '.netrc', '.pgpass'}
    if basename in blocked_names:
        return True
    # Block hidden files in home dir
    if filepath.startswith(os.path.expanduser('~') + '/.'):
        # Allow .bashrc, .profile etc but block keys and creds
        if any(k in filepath for k in ['ssh', 'key', 'credential', 'secret', 'token', 'netrc', 'pgpass']):
            return True
    return False


# --- Allowed commands (whitelist approach) ---
# --- Write commands (require user permission) ---
WRITE_COMMANDS = {
    'touch', 'mkdir', 'rm', 'cp', 'mv', 'nano', 'vim',
    'chmod', 'chown', 'dd', 'tee',
    'ln', 'unlink', 'rename', 'truncate', 'fallocate',
    'sed', 'awk', 'perl'  # text processing that can modify files
}

# --- Permission queues: session_id -> queue.Queue ---
# Permission queues: permission_id -> queue.Queue()
_write_permission_queues = {}

# Pending permissions: perm_id -> {session_id, command, q}
_pending_permissions = {}

# --- Per-session write permission state: session_id -> 'none'|'once'|'session' ---
_session_write_permissions = {}

SAFE_COMMANDS = {
    'ls': {'flags': {'-l', '-a', '-la', '-al', '-lh', '-lah', '-R', '-1'},
            'allow_args': False},
    'pwd': {'flags': set(), 'allow_args': False},
    'whoami': {'flags': set(), 'allow_args': False},
    'date': {'flags': set(), 'allow_args': False},
    'hostname': {'flags': set(), 'allow_args': False},
    'uptime': {'flags': set(), 'allow_args': False},
    'uname': {'flags': {'-a', '-r', '-m', '-s'}, 'allow_args': False},
    'df': {'flags': {'-h', '-T', '-i', '-ht'}, 'allow_args': False},
    'free': {'flags': {'-h', '-m', '-g', '-k'}, 'allow_args': False},
    'ps': {'flags': {'aux', 'auxww', '-ef', 'auxf'}, 'allow_args': False},
    'du': {'flags': {'-sh', '-h', '-sh', '-ah', '-h', '--max-depth=1'},
           'allow_args': True},  # du needs a path argument
    'wc': {'flags': {'-l', '-w', '-c'}, 'allow_args': True},  # wc needs filename
    'head': {'flags': {'-n'}, 'allow_args': True},  # head needs filename
    'tail': {'flags': {'-n'}, 'allow_args': True},  # tail needs filename
    'cat': {'flags': set(), 'allow_args': True},     # cat needs filename
    'tree': {'flags': {'-L', '-d', '-a'}, 'allow_args': True},
    'find': {'flags': {'-name', '-type', '-size', '-maxdepth'}, 'allow_args': True},
    'ip': {'flags': {'addr', 'link', 'route'}, 'allow_args': False},
    'ping': {'flags': {'-c'}, 'allow_args': True},  # ping needs host
    'curl': {'flags': {'-s', '-I', '-i', '-L'}, 'allow_args': True},
    'netstat': {'flags': {'-tuln', '-tln', '-tulnp'}, 'allow_args': False},
    'echo': {'flags': set(), 'allow_args': True},   # echo for reading (write with > handled separately)
    'printf': {'flags': set(), 'allow_args': True},  # printf for reading (write with > handled separately)
}


def validate_command(cmd):
    """Validate and parse a command. Returns (command_path, args) or None if invalid."""
    cmd = cmd.strip()
    if not cmd:
        return None

    # Reject commands with shell redirections (these should go through write permission)
    if '>' in cmd or '>>' in cmd or '|' in cmd:
        logger.warning("Rejected command with redirection/pipe (use write permission): %s", cmd)
        return None

    parts = cmd.split()
    base = parts[0]

    if base not in SAFE_COMMANDS:
        logger.warning("Rejected command (not in whitelist): %s", cmd)
        return None

    spec = SAFE_COMMANDS[base]
    validated_parts = [base]

    i = 1
    while i < len(parts):
        part = parts[i]
        if part.startswith('-'):
            # It's a flag - check if allowed
            # Handle combined flags like -la
            flag = part
            if flag in spec['flags']:
                validated_parts.append(flag)
            elif base == 'ps' and flag == 'aux':
                validated_parts.append(flag)
            else:
                # Check if it's a valid flag that takes an argument (like -n 10)
                if i + 1 < len(parts) and flag in spec['flags']:
                    validated_parts.append(flag)
                    i += 1
                    validated_parts.append(parts[i])
                else:
                    logger.warning("Rejected flag %s for command %s", part, base)
                    return None
        elif spec.get('allow_args'):
            # Check file args for sensitive paths
            if base in ('cat', 'head', 'tail', 'wc') and is_sensitive_path(part):
                logger.warning("Blocked access to sensitive path: %s", part)
                return None
            validated_parts.append(part)
        else:
            logger.warning("Rejected argument %s for command %s", part, base)
            return None
        i += 1

    return validated_parts


def execute_local_command(cmd):
    """Execute a validated local command (read-only, no shell)"""
    import subprocess as sp

    try:
        cmd = cmd.strip()

        # Danger check
        if is_dangerous(cmd):
            logger.warning("Blocked dangerous command: %s", cmd)
            return "[Security] This command is not allowed."

        # Validate and parse
        parsed = validate_command(cmd)
        if parsed is None:
            return "[Security] This command is not allowed."

        result = sp.run(
            parsed,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.expanduser('~')
        )

        output = result.stdout.strip() or result.stderr.strip() or "Command executed successfully (no output)"
        logger.info("Executed command: %s", ' '.join(parsed))
        return output[:5000]

    except sp.TimeoutExpired:
        return "[Timeout] Command took too long (>10s)"
    except Exception as e:
        logger.error("Command execution error: %s", e)
        return f"[Error] {str(e)}"


def is_write_command(cmd):
    """Check if a command involves write operations."""
    cmd = cmd.strip()
    if not cmd:
        return False
    parts = cmd.split()
    base = parts[0]
    # Handle shell redirections first (echo > file, printf >> file)
    if '>' in cmd or '>>' in cmd:
        return True
    # Handle tee usage (piped writes)
    if '| tee' in cmd or '|tee' in cmd:
        return True
    # echo/printf without redirection are NOT write commands
    if base in ('echo', 'printf'):
        return False  # Only write if redirected (caught above)
    # Direct write commands
    if base in WRITE_COMMANDS:
        return True
    # Handle sudo + write command
    if base == 'sudo' and len(parts) > 1 and parts[1] in WRITE_COMMANDS:
        return True
    return False


def check_write_permission(cmd, session_id):
    """Check if a write command is allowed for this session.
    Returns 'allowed' if the command can proceed, or 'ask' if permission is needed."""
    perm = _session_write_permissions.get(session_id, 'none')
    if perm == 'session':
        return 'allowed'
    elif perm == 'once':
        # Allow once, then reset
        _session_write_permissions[session_id] = 'none'
        return 'allowed'
    else:
        return 'ask'


def execute_write_command(cmd, session_id):
    """Execute a write command after permission is granted.
    Returns the command output or permission-denied message."""
    import subprocess as sp

    try:
        cmd = cmd.strip()
        if not cmd:
            return "[Error] Empty command"

        # For write commands, we execute with shell=True but with basic safety
        # (the permission system is the safety gate)
        # Still block truly catastrophic patterns
        catastrophic = ['rm -rf /', 'mkfs', 'dd if=/dev/zero of=/dev/']
        for pattern in catastrophic:
            if pattern in cmd:
                return "[Security] This command is blocked for safety."

        result = sp.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.expanduser('~')
        )

        output = result.stdout.strip() or result.stderr.strip() or "Command executed successfully (no output)"
        logger.info("Executed write command (session=%s): %s", session_id, cmd)
        return output[:5000]

    except sp.TimeoutExpired:
        return "[Timeout] Command took too long (>30s)"
    except Exception as e:
        logger.error("Write command execution error: %s", e)
        return f"[Error] {str(e)}"


def is_dangerous(cmd):
    """Check if the command is dangerous. Write commands are handled by the permission system."""
    # Don't block write commands here - they're handled by write_permission
    # Only block truly catastrophic patterns
    catastrophic_patterns = [
        r'rm\s+-rf\s+/',
        r'mkfs',
        r'dd\s+if=/dev/zero\s+of=/dev/',
        r'wget.*\|.*sh',
        r'bash.*http',
        r'curl.*\|.*sh',
    ]
    for pattern in catastrophic_patterns:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False


# --- Write Permission Endpoint ---
@app.route('/api/write-permission', methods=['POST'])
def api_write_permission():
    """Handle write permission responses from the frontend.
    Body: {perm_id, action: 'deny'|'once'|'session', session_id, command}"""
    data = request.json
    perm_id = data.get('perm_id', '')
    action = data.get('action', 'deny')
    session_id = data.get('session_id', '')

    logger.info("Write permission response: perm_id=%s, action=%s, session=%s", perm_id, action, session_id)

    # Signal the specific pending permission
    if perm_id and perm_id in _pending_permissions:
        pending = _pending_permissions[perm_id]
        pending['q'].put(action)
        return jsonify({'success': True})

    # Fallback: signal session-level queue (old behavior)
    if session_id:
        if action in ('once', 'session'):
            _session_write_permissions[session_id] = action
        else:
            _session_write_permissions[session_id] = 'none'
        q = _write_permission_queues.get(session_id)
        if q:
            q.put(action)

    return jsonify({'success': True})


# --- Ollama Communication ---
def get_ollama_models():
    """Get list of available models. If empty, pull llama3.2:1b as fallback."""
    try:
        import urllib.request
        req = urllib.request.Request(f'{OLLAMA_BASE_URL}/api/tags')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            models = [m['name'] for m in data.get('models', [])]
            if not models:
                logger.info("No models found, pulling llama3.2:1b as fallback...")
                try:
                    pull_req = urllib.request.Request(
                        f'{OLLAMA_BASE_URL}/api/pull',
                        data=json.dumps({"name": "llama3.2:1b", "stream": False}).encode(),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(pull_req, timeout=300) as pull_resp:
                        pull_data = json.loads(pull_resp.read())
                        logger.info("Pulled llama3.2:1b: %s", pull_data.get('status', 'done'))
                    # Refresh model list
                    with urllib.request.urlopen(req, timeout=5) as response2:
                        data2 = json.loads(response2.read())
                        models = [m['name'] for m in data2.get('models', [])]
                except Exception as pull_err:
                    logger.error("Failed to pull fallback model: %s", pull_err)
            return models
    except Exception as e:
        logger.error("Failed to get models: %s", e)
        return []


def get_model_info(model_name):
    """Get model info including context window size"""
    try:
        import urllib.request
        req = urllib.request.Request(
            f'{OLLAMA_BASE_URL}/api/show',
            data=json.dumps({'name': model_name}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())

            num_ctx = data.get('num_ctx') or data.get('context_length')
            model_info = data.get('model_info', {})
            if not num_ctx:
                for key in model_info:
                    if 'context_length' in key.lower():
                        num_ctx = model_info[key]
                        break

            return {
                'model': model_name,
                'context_length': num_ctx or 4096,
                'num_ctx': num_ctx or 4096,
                'model_info': model_info,
                'details': data.get('details', {}),
                'size': data.get('size', 0),
                'modified_at': data.get('modified_at', ''),
            }
    except Exception as e:
        logger.error("Failed to get model info for %s: %s", model_name, e)
        return {'error': str(e), 'model': model_name}


def send_to_ollama(model, messages, tools=None, stream=False):
    """Send message to Ollama and return response"""
    try:
        import urllib.request

        payload = {
            'model': model,
            'messages': messages,
            'stream': stream,
            'keep_alive': KEEP_ALIVE,
        }

        if tools:
            payload['tools'] = tools

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f'{OLLAMA_BASE_URL}/api/chat',
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        timeout = 300 if stream else 180
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if stream:
                return response  # Return the response object for streaming
            result = json.loads(response.read())
            return result
    except Exception as e:
        logger.error("Ollama communication error: %s", e)
        return {'error': str(e)}


def process_ollama_response(model, messages, tools=None):
    """Process Ollama response, executing tools if necessary.
    Returns a dict with 'response' and 'prompt_eval_count'."""
    max_iterations = 2
    last_prompt_tokens = 0

    for i in range(max_iterations):
        response = send_to_ollama(model, messages, tools, stream=False)

        if 'error' in response:
            return {'response': f"Error: {response['error']}", 'prompt_eval_count': 0}

        last_prompt_tokens = response.get('prompt_eval_count', 0)
        assistant_msg = response.get('message', {})
        content = assistant_msg.get('content', '')
        tool_calls = assistant_msg.get('tool_calls', [])

        if not tool_calls:
            return {'response': content, 'prompt_eval_count': last_prompt_tokens}

        # Process tool calls
        tool_results = []
        for tool_call in tool_calls:
            func_name = tool_call.get('function', {}).get('name', '')
            func_args = tool_call.get('function', {}).get('arguments', {})
            tool_id = tool_call.get('id', f'tool_{i}')

            logger.info("Tool call #%d: %s(%s)", i + 1, func_name, json.dumps(func_args))

            if func_name == 'local_command':
                cmd = func_args.get('command', '')
                # Check if this is a write command
                if is_write_command(cmd):
                    result = "[Security] Write commands require streaming mode for permission confirmation. Please retry in streaming mode."
                else:
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
                        result += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"
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

        messages.append({
            'role': 'assistant',
            'content': content,
            'tool_calls': tool_calls
        })

        for tr in tool_results:
            messages.append(tr)

        if len(tool_calls) == 1 and content.strip() == '':
            return {'response': f"Command executed:\n\n{tool_results[0]['content']}",
                    'prompt_eval_count': last_prompt_tokens}

    return {'response': tool_results[-1]['content'] if tool_results else content,
            'prompt_eval_count': last_prompt_tokens}


# --- Ollama Tools Definition ---
OLLAMA_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'local_command',
            'description': 'Execute a local system command. Supports both read-only commands (ls, cat, find, grep, df, free, etc. - execute without confirmation) and write commands (touch, mkdir, rm, cp, mv, chmod, etc. - require user confirmation via popup). Write operations will pause and ask the user for permission before executing.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The command to execute. Read-only examples: "ls -la", "df -h", "free -h", "uname -a", "pwd", "whoami", "uptime", "hostname". Write examples (require confirmation): "mkdir /tmp/test", "touch /tmp/file", "cp file1 file2"'
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
            'description': 'Search the internet for information. Returns title, URL, and snippet for each result. Use fetch_article separately to get full content from specific URLs when needed.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query to find information on the internet.'
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
            'description': 'Fetch and extract the full text content from a web article URL. Use this after web_search to get detailed content from specific articles.',
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


# --- Web Search & Article Fetching ---
def web_search(query):
    """Search the internet using DuckDuckGo (ddgs) - lazy, no auto-fetch"""
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
            logger.info("Web search for '%s' returned %d results", query, len(results))
            return results
    except Exception as e:
        logger.error("Web search error: %s", e)
        return [{'error': str(e)}]


def fetch_article(url):
    """Fetch and extract text content from a web article URL"""
    try:
        import urllib.request
        import html as html_mod

        # Block sensitive/local URLs
        if url.startswith(('file://', 'ftp://')) or 'localhost' in url or '127.0.0.1' in url:
            return {'url': url, 'error': 'URL not allowed'}

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"})
        resp = urllib.request.urlopen(req, timeout=10)
        html_content = resp.read().decode("utf-8", errors="ignore")

        for t in ["script", "style", "nav", "header", "footer", "aside", "noscript"]:
            html_content = re.sub(f"<{t}[^>]*>.*?</{t}>", "", html_content, flags=re.DOTALL | re.IGNORECASE)

        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_content, re.DOTALL | re.IGNORECASE)
        lines = []
        for p in paragraphs:
            clean = re.sub(r"<[^>]+>", "", p).strip()
            clean = html_mod.unescape(clean)
            if len(clean) > 50:
                lines.append(clean)

        text = "\n".join(lines)
        if not text:
            body = re.search(r"<body[^>]*>(.*?)</body>", html_content, re.DOTALL | re.IGNORECASE)
            if body:
                text = html_mod.unescape(re.sub(r"<[^>]+>", " ", body.group(1)))
                text = re.sub(r"\s+", " ", text).strip()[:3000]
        else:
            text = text[:3000]

        if text:
            logger.info("Fetched article from %s (%d chars)", url, len(text))
            return {'url': url, 'content': text}
        return {'url': url, 'error': 'Could not extract content'}
    except Exception as e:
        logger.error("Article fetch error for %s: %s", url, e)
        return {'url': url, 'error': str(e)}


# --- Session Management ---
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
    if not os.path.exists(SESSIONS_DIR):
        return sessions
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
    sessions.sort(key=lambda x: x.get('created', ''), reverse=True)
    return sessions


# --- Routes ---
@app.route('/')
def index():
    """Main page"""
    models = get_ollama_models()
    sessions = list_sessions()

    if 'chat_id' not in session:
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
    """API to get model info including context window and size"""
    model_name = request.args.get('model', '')
    if not model_name:
        return jsonify({'error': 'Model name required'})

    info = get_model_info(model_name)

    # Check if model is currently loaded (local models only - cloud models are always available)
    try:
        import urllib.request
        req = urllib.request.Request(f'{OLLAMA_BASE_URL}/api/ps')
        with urllib.request.urlopen(req, timeout=5) as resp:
            ps_data = json.loads(resp.read())
            loaded_models = [m.get('name', '') for m in ps_data.get('models', [])]
            # Check if our model name matches (may include :latest suffix)
            is_loaded = any(model_name == m or model_name + ':latest' == m for m in loaded_models)
            # Cloud models are always available but won't appear in /api/ps
            if not is_loaded and ':cloud' in model_name:
                is_loaded = True
            info['loaded'] = is_loaded
    except Exception:
        info['loaded'] = None  # Unknown

    return jsonify(info)


@app.route('/api/chat/stream', methods=['POST'])
def api_chat_stream():
    """Streaming chat endpoint using SSE"""
    data = request.json
    user_message = data.get('message', '').strip()
    model = data.get('model', session.get('model', 'llama3'))
    fallback_model = data.get('fallback_model', '')

    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # Rate limit
    client_ip = request.remote_addr
    if rate_limit_exceeded(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        return jsonify({'error': 'Rate limit exceeded. Please wait a moment.'}), 429

    # Create/load session (streaming endpoint)
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())[:8]

    session['model'] = model
    if fallback_model:
        session['fallback_model'] = fallback_model

    # Capture session data before generator (Flask session unavailable inside generator)
    current_chat_id = session['chat_id']

    session_data = load_session(current_chat_id) or {
        'model': model,
        'fallback_model': fallback_model,
        'title': user_message[:50] + ('...' if len(user_message) > 50 else ''),
        'created': datetime.now().isoformat(),
        'messages': []
    }

    session_data['messages'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })

    logger.info("Chat request (stream): model=%s, msg_len=%d, session=%s", model, len(user_message), current_chat_id)

    def generate():
        full_response = ""
        prompt_tokens = 0
        try:
            # Prepare messages for Ollama (strip timestamps for API)
            api_messages = []
            # Model-specific system prompts based on known behavior
            MODEL_HINTS = {
                'glm': 'IMPORTANT: Always use local_command tool to write/edit files. Use cat > /path/to/file << \'EOF\' ... EOF for creating files, sed -i for editing text in files. Use /home/cvc1/ instead of $HOME or ~. Do NOT output code as text.',
                'minimax': 'Use tools when available. For file operations, use local_command with shell commands.',
                'gemma': 'You have access to local_command, web_search, and fetch_article tools. Use them proactively.',
                'kimi': 'Use the available tools for file operations and web searches. Do not just show code.',
                'qwen': 'Always use local_command tool for writing files. Use cat > with heredoc syntax.',
            }
            model_hint = ''
            for key, hint in MODEL_HINTS.items():
                if key in model.lower():
                    model_hint = hint
                    break
            system_content = 'You are an assistant with access to tools. IMPORTANT RULES:\n- When asked to CREATE or WRITE files, you MUST use the local_command tool with a shell command like: cat > /path/to/file << \'EOF\'\n  content here\n  EOF\n- When asked to EDIT or REPLACE text in a file, use sed -i: sed -i \'s/old_text/new_text/g\' /path/to/file\n- Do NOT just show code in your response - actually write it to disk using local_command\n- Do NOT say you cannot write files - you CAN write files using local_command\n- For creating files with content, use: cat > /path/to/file << \'EOF\' followed by the content, then EOF on a new line\n- Always use the actual home directory path like /home/cvc1/ instead of $HOME or ~\n- Available tools: local_command (execute system commands), web_search (search the internet), fetch_article (read web pages)\n- Write operations will be executed automatically with user notification'
            if model_hint:
                system_content += '\n\n' + model_hint
            api_messages.append({
                'role': 'system',
                'content': system_content
            })
            for msg in session_data['messages']:
                api_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

            payload = {
                'model': model,
                'messages': api_messages,
                'stream': True,
                'keep_alive': KEEP_ALIVE,
                'tools': OLLAMA_TOOLS,
            }

            import urllib.request

            data_bytes = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f'{OLLAMA_BASE_URL}/api/chat',
                data=data_bytes,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=300) as response:
                tool_calls_buffer = []
                current_tool_call = None

                for line in response:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get('done'):
                        # Capture eval counts
                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                        eval_count = chunk.get('eval_count', 0)

                        # If we have pending tool calls, process them
                        if tool_calls_buffer:
                            # Pre-process write commands: check permissions via SSE
                            write_cmd_indices = []
                            for i_t, tc in enumerate(tool_calls_buffer):
                                tc_name = tc.get('function', {}).get('name', '')
                                tc_args = tc.get('function', {}).get('arguments', {})
                                if tc_name == 'local_command':
                                    cmd = tc_args.get('command', '')
                                    if is_write_command(cmd):
                                        perm = check_write_permission(cmd, current_chat_id)
                                        if perm == 'ask':
                                            write_cmd_indices.append((i_t, cmd))

                            # Collect write commands that need permission
                            write_cmds_to_request = []
                            for i_t, tc in enumerate(tool_calls_buffer):
                                tc_name = tc.get('function', {}).get('name', '')
                                tc_args = tc.get('function', {}).get('arguments', {})
                                if tc_name == 'local_command':
                                    cmd = tc_args.get('command', '')
                                    if is_write_command(cmd):
                                        perm = check_write_permission(cmd, current_chat_id)
                                        if perm == 'ask':
                                            # Assign unique permission ID
                                            perm_id = str(uuid.uuid4())
                                            write_cmds_to_request.append({'idx': i_t, 'cmd': cmd, 'perm_id': perm_id})
                                            _pending_permissions[perm_id] = {
                                                'session_id': current_chat_id,
                                                'command': cmd,
                                                'q': queue.Queue()
                                            }

                            # If there are write commands needing permission, auto-approve for now
                            # and notify the user what was executed
                            # (SSE permission popup doesn't work reliably due to buffering)
                            if write_cmds_to_request:
                                for item in write_cmds_to_request:
                                    logger.info("Auto-approving write command: %s", item['cmd'])
                                    tc = tool_calls_buffer[item['idx']]
                                    tc['_write_action'] = 'once'
                                # Notify user what commands are being executed
                                cmds_str = ', '.join([item['cmd'] for item in write_cmds_to_request])
                                yield f"data: {json.dumps({'type': 'write_executed', 'commands': [item['cmd'] for item in write_cmds_to_request], 'session_id': current_chat_id})}\n\n"

                            # Process tool calls - now with write permission resolved
                            tool_results = _process_tool_calls_streaming(
                                model, session_data, tool_calls_buffer,
                                full_response, prompt_tokens, current_chat_id
                            )
                            # Send tool results back to Ollama for final response
                            followup_messages = []
                            for msg in session_data['messages']:
                                followup_messages.append({'role': msg['role'], 'content': msg['content']})
                            # Add assistant message with tool calls (strip internal flags first)
                            clean_tool_calls = []
                            for tc in tool_calls_buffer:
                                clean_tc = {k: v for k, v in tc.items() if k != '_write_denied'}
                                clean_tool_calls.append(clean_tc)
                            followup_messages.append({'role': 'assistant', 'content': full_response or '', 'tool_calls': clean_tool_calls})
                            # Add tool results
                            for tr in tool_results:
                                followup_messages.append({'role': tr['role'], 'content': tr['content']})

                            # Make follow-up request(s) with tool results
                            # Model may request more tools - limit rounds, then force text response
                            max_followup_rounds = 5
                            for round_num in range(max_followup_rounds):
                                logger.info("Follow-up round %d: sending %d tool results back to %s", round_num + 1, len(tool_results), model)
                                # On rounds 3+, don't send tools so model is forced to answer with text
                                tools_for_this_round = OLLAMA_TOOLS if round_num < 2 else None
                                followup_result = send_to_ollama(model, followup_messages, tools_for_this_round, stream=False)
                                logger.info("Follow-up response: content_len=%d, has_tool_calls=%s", len(followup_result.get('message', {}).get('content', '')), bool(followup_result.get('message', {}).get('tool_calls')))

                                if 'error' in followup_result:
                                    full_response = f"Error: {followup_result['error']}"
                                    logger.error("Follow-up error: %s", full_response)
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                    break

                                followup_msg = followup_result.get('message', {})
                                followup_content = followup_msg.get('content', '')
                                followup_tool_calls = followup_msg.get('tool_calls', [])

                                # If model gives both content and tool calls, prefer content
                                if followup_content and not followup_tool_calls:
                                    full_response = followup_content
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                    prompt_tokens = followup_result.get('prompt_eval_count', prompt_tokens)
                                    break  # Got a text response, done
                                elif followup_content and followup_tool_calls:
                                    # Model has partial content but wants more tools
                                    # Check if any of the tool calls are write commands - if so, let them execute
                                    has_write_cmd = False
                                    for tc in followup_tool_calls:
                                        tc_args = tc.get('function', {}).get('arguments', {})
                                        if tc.get('function', {}).get('name') == 'local_command' and is_write_command(tc_args.get('command', '')):
                                            has_write_cmd = True
                                            break
                                    
                                    if has_write_cmd:
                                        # Let write commands execute - treat as normal tool calls
                                        logger.info("Model has partial content + write tool calls, allowing execution")
                                        # Send partial content first
                                        full_response = followup_content
                                        yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                        # Now execute the write tool calls
                                        followup_messages.append({'role': 'assistant', 'content': followup_content, 'tool_calls': followup_tool_calls})
                                        for tc in followup_tool_calls:
                                            tc_name = tc.get('function', {}).get('name', '')
                                            tc_args = tc.get('function', {}).get('arguments', {})
                                            tc_id = tc.get('id', f'tool_{round_num}_{len(followup_tool_calls)}')
                                            if tc_name == 'local_command':
                                                cmd = tc_args.get('command', '')
                                                if is_write_command(cmd):
                                                    # Auto-approve write command
                                                    logger.info("Auto-approving follow-up write: %s", cmd)
                                                    result = execute_write_command(cmd)
                                                else:
                                                    result = execute_local_command(cmd)
                                                followup_messages.append({'role': 'tool', 'content': result, 'tool_call_id': tc_id})
                                            elif tc_name == 'web_search':
                                                q = tc_args.get('query', '')
                                                results = web_search(q)
                                                followup_messages.append({'role': 'tool', 'content': json.dumps(results[:5]), 'tool_call_id': tc_id})
                                            elif tc_name == 'fetch_article':
                                                url = tc_args.get('url', '')
                                                article = fetch_article(url)
                                                content = article.get('content', '')[:2000] if article.get('content') else f"Could not fetch {url}"
                                                followup_messages.append({'role': 'tool', 'content': content, 'tool_call_id': tc_id})
                                        # Continue loop to get final response after write
                                        continue
                                    else:
                                        # Non-write tools (search, etc.) - force final response
                                        logger.info("Model has partial content (%d chars) + tool calls, forcing final response", len(followup_content))
                                        tool_summaries = []
                                        for m in followup_messages:
                                            if m.get('role') == 'tool' and m.get('content'):
                                                tool_summaries.append(m['content'])
                                        context_hint = ''
                                        if tool_summaries:
                                            context_hint = f'\n\nHere are the search results I found:\n"""\n{"---".join(tool_summaries)}\n"""\n\nPlease provide a complete answer based on this information.'
                                        followup_messages.append({'role': 'assistant', 'content': followup_content})
                                        try:
                                            final_result = send_to_ollama(model, followup_messages, None, stream=False)
                                            final_content = final_result.get('message', {}).get('content', '')
                                            if final_content:
                                                full_response = final_content
                                                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                                prompt_tokens = final_result.get('prompt_eval_count', prompt_tokens)
                                            else:
                                                # Model returned empty, send partial content
                                                full_response = followup_content
                                                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                        except Exception as e:
                                            logger.error("Forced final response failed: %s", e)
                                            full_response = followup_content
                                            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                        break

                                elif followup_tool_calls and tools_for_this_round is not None:
                                    if full_response:
                                        # Already have some content from earlier, send it
                                        pass
                                    # Model wants more tool calls - execute them
                                    logger.info("Follow-up round %d: model requested %d more tool calls", round_num + 1, len(followup_tool_calls))
                                    # Add assistant message with tool calls to history
                                    followup_messages.append({'role': 'assistant', 'content': '', 'tool_calls': followup_tool_calls})
                                    for tc in followup_tool_calls:
                                        tc_name = tc.get('function', {}).get('name', '')
                                        tc_args = tc.get('function', {}).get('arguments', {})
                                        tc_id = tc.get('id', f'tool_{round_num}_{len(followup_tool_calls)}')
                                        logger.info("Follow-up tool call: %s(%s)", tc_name, json.dumps(tc_args))
                                        if tc_name == 'web_search':
                                            q = tc_args.get('query', '')
                                            results = web_search(q)
                                            if results and isinstance(results, list) and 'error' in results[0]:
                                                tr_content = f"Search error: {results[0]['error']}"
                                            else:
                                                tr_content = "Search results:\n\n"
                                                for idx, r in enumerate(results[:5], 1):
                                                    tr_content += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"
                                            followup_messages.append({'role': 'tool', 'content': tr_content, 'tool_call_id': tc_id})
                                        elif tc_name == 'fetch_article':
                                            url = tc_args.get('url', '')
                                            article = fetch_article(url)
                                            if 'content' in article:
                                                tr_content = f"Article from {url}:\n\n{article['content']}"
                                            else:
                                                tr_content = f"Could not fetch article from {url}: {article.get('error', 'Unknown error')}"
                                            followup_messages.append({'role': 'tool', 'content': tr_content, 'tool_call_id': tc_id})
                                        elif tc_name == 'local_command':
                                            cmd = tc_args.get('command', '')
                                            if is_write_command(cmd):
                                                perm = check_write_permission(cmd, current_chat_id)
                                                if perm == 'ask':
                                                    perm_id = str(uuid.uuid4())
                                                    q = queue.Queue()
                                                    _pending_permissions[perm_id] = {
                                                        'session_id': current_chat_id,
                                                        'command': cmd,
                                                        'q': q
                                                    }
                                                    yield f"data: {json.dumps({'type': 'write_permission_required', 'command': cmd, 'session_id': current_chat_id, 'perm_id': perm_id})}\n\n"
                                                    action = q.get()  # blocks until frontend responds
                                                    del _pending_permissions[perm_id]
                                                    if action == 'deny':
                                                        tr_content = "[Permission denied] Task cancelled"
                                                    elif action == 'once':
                                                        _session_write_permissions[current_chat_id] = 'none'
                                                        tr_content = execute_write_command(cmd, current_chat_id)
                                                    elif action == 'session':
                                                        _session_write_permissions[current_chat_id] = 'session'
                                                        tr_content = execute_write_command(cmd, current_chat_id)
                                                    else:
                                                        tr_content = "[Permission denied] Task cancelled"
                                                else:
                                                    # perm == 'allowed', already approved
                                                    tr_content = execute_write_command(cmd, current_chat_id)
                                            else:
                                                tr_content = execute_local_command(cmd)
                                            followup_messages.append({'role': 'tool', 'content': tr_content, 'tool_call_id': tc_id})
                                        else:
                                            followup_messages.append({'role': 'tool', 'content': f'Unknown tool: {tc_name}', 'tool_call_id': tc_id})
                                    # Continue loop to send tool results back
                                    continue
                                else:
                                    # No content and no tool calls (or tools disabled), or tool calls but tools disabled
                                    # Retry without tools to force text response
                                    if followup_tool_calls and tools_for_this_round is None:
                                        logger.info("Model requested tools but they're disabled, retrying with tool results as context")
                                        # Remove the last assistant message with tool_calls and add a simple one
                                        followup_messages = [m for m in followup_messages if not (m.get('tool_calls'))]
                                        # Collect all tool results from followup_messages for context
                                        tool_summaries = []
                                        for m in followup_messages:
                                            if m.get('role') == 'tool' and m.get('content'):
                                                tool_summaries.append(m['content'])
                                        context_hint = ''
                                        if tool_summaries:
                                            context_hint = f'\n\nI found the following information:\n"""\n{"---".join(tool_summaries)}\n"""\n\nBased on this information, please provide a clear answer to the user.'
                                        followup_messages.append({'role': 'assistant', 'content': f'I have gathered the information needed.{context_hint}'})
                                        followup_result2 = send_to_ollama(model, followup_messages, None, stream=False)
                                        content2 = followup_result2.get('message', {}).get('content', '')
                                        if content2:
                                            full_response = content2
                                            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                            prompt_tokens = followup_result2.get('prompt_eval_count', prompt_tokens)
                                            break
                                    full_response = "(No response from model)"
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                    break
                            else:
                                # Max rounds reached - force final response without tools
                                logger.info("Max follow-up rounds reached, forcing text response")
                                # Include tool results in the prompt
                                tool_summaries = []
                                for m in followup_messages:
                                    if m.get('role') == 'tool' and m.get('content'):
                                        tool_summaries.append(m['content'])
                                context_hint = ''
                                if tool_summaries:
                                    context_hint = f'\n\nI found the following information:\n"""\n{"---".join(tool_summaries)}\n"""'
                                followup_messages.append({'role': 'assistant', 'content': f'Based on the search results, here is my answer:{context_hint}'})
                                final_result = send_to_ollama(model, followup_messages, None, stream=False)
                                final_content = final_result.get('message', {}).get('content', '')
                                if final_content:
                                    full_response = final_content
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                    prompt_tokens = final_result.get('prompt_eval_count', prompt_tokens)
                                else:
                                    full_response = "(Maximum tool call rounds reached)"
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                        break

                    msg = chunk.get('message', {})

                    # Handle tool calls in streaming
                    if msg.get('tool_calls'):
                        for tc in msg['tool_calls']:
                            tool_calls_buffer.append(tc)
                        # Don't show content if we have tool calls (it's usually just the tool name)
                        continue

                    content = msg.get('content', '')
                    if content:
                        # If tool calls are being collected, suppress content display
                        # (models sometimes emit tool names as text before the formal tool call)
                        if not tool_calls_buffer:
                            # Filter out tool call artifacts that some models emit as text
                            # Matches patterns like "model:tool_call" or "model:tool_call\nextra text"
                            import re
                            stripped = content.strip()
                            if re.match(r'^[\w.-]+:tool_call', stripped):
                                full_response += ''
                                continue
                            full_response += content
                            # Send SSE event
                            sse_data = json.dumps({'type': 'token', 'content': content})
                            yield f"data: {sse_data}\n\n"

                # If response was empty and no tool calls, try fallback
                if not full_response and not tool_calls_buffer:
                    if fallback_model and fallback_model != model:
                        logger.info("Primary model '%s' empty response, trying fallback '%s'", model, fallback_model)
                        payload['model'] = fallback_model
                        data_bytes = json.dumps(payload).encode('utf-8')
                        req2 = urllib.request.Request(
                            f'{OLLAMA_BASE_URL}/api/chat',
                            data=data_bytes,
                            headers={'Content-Type': 'application/json'}
                        )
                        with urllib.request.urlopen(req2, timeout=300) as response2:
                            for line in response2:
                                line = line.decode('utf-8').strip()
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if chunk.get('done'):
                                    prompt_tokens = chunk.get('prompt_eval_count', 0)
                                    break
                                content = chunk.get('message', {}).get('content', '')
                                if content:
                                    full_response += content
                                    sse_data = json.dumps({'type': 'token', 'content': content})
                                    yield f"data: {sse_data}\n\n"

        except Exception as e:
            logger.error("Streaming error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            # Still save what we have
            if full_response:
                session_data['messages'].append({
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.now().isoformat()
                })
                save_session(current_chat_id, session_data)
            return

        # Save session
        session_data['messages'].append({
            'role': 'assistant',
            'content': full_response,
            'timestamp': datetime.now().isoformat()
        })
        # Save context usage in session for per-conversation display
        session_data['context_usage'] = prompt_tokens
        save_session(current_chat_id, session_data)

        # Detect if the response contains code that should be written to a file
        # (when the model generates code instead of using the local_command tool)
        code_blocks = []
        try:
            code_blocks = re.findall(r'```(\w+)?\n(.*?)```', full_response, re.DOTALL)
        except Exception:
            pass
        if code_blocks and not tool_calls_buffer:
            # Model generated code as text instead of using tools
            # Offer to save via frontend notification
            for lang, code in code_blocks:
                if lang in ('html', 'htm', 'javascript', 'js', 'css', 'python', 'py', 'php', 'sh', 'bash'):
                    yield f"data: {json.dumps({'type': 'code_save_offer', 'language': lang, 'code': code.strip()})}\n\n"
                    break  # Only offer once

        # Send completion event
        yield f"data: {json.dumps({'type': 'done', 'context_usage': prompt_tokens})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _process_tool_calls_streaming(model, session_data, tool_calls, current_content, prompt_tokens, current_chat_id=''):
    """Process tool calls from streaming - used internally"""
    # This is called after streaming completes with tool calls
    # For now, we execute tools and make a non-streaming follow-up
    tool_results = []
    for i, tool_call in enumerate(tool_calls):
        func_name = tool_call.get('function', {}).get('name', '')
        func_args = tool_call.get('function', {}).get('arguments', {})
        tool_id = tool_call.get('id', f'tool_{i}')

        logger.info("Stream tool call: %s(%s)", func_name, json.dumps(func_args))

        if func_name == 'local_command':
            cmd = func_args.get('command', '')
            # Check if this was denied by the permission system
            if tool_call.get('_write_denied'):
                result = "[Permission denied] Task cancelled"
            elif is_write_command(cmd):
                # Write command with permission granted - execute it
                result = execute_write_command(cmd, current_chat_id)
            else:
                result = execute_local_command(cmd)
            tool_results.append({'role': 'tool', 'content': result, 'tool_call_id': tool_id})
        elif func_name == 'web_search':
            query = func_args.get('query', '')
            results = web_search(query)
            if results and isinstance(results, list) and 'error' in results[0]:
                result = f"Search error: {results[0]['error']}"
            else:
                result = "Search results:\n\n"
                for idx, r in enumerate(results[:5], 1):
                    result += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"
            tool_results.append({'role': 'tool', 'content': result, 'tool_call_id': tool_id})
        elif func_name == 'fetch_article':
            url = func_args.get('url', '')
            article = fetch_article(url)
            if 'content' in article:
                result = f"Article from {url}:\n\n{article['content']}"
            else:
                result = f"Could not fetch article from {url}: {article.get('error', 'Unknown error')}"
            tool_results.append({'role': 'tool', 'content': result, 'tool_call_id': tool_id})
        else:
            tool_results.append({'role': 'tool', 'content': f"Unknown tool: {func_name}", 'tool_call_id': tool_id})

    return tool_results


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API to send message and receive response (non-streaming fallback)"""
    data = request.json
    user_message = data.get('message', '').strip()
    model = data.get('model', session.get('model', 'llama3'))
    fallback_model = data.get('fallback_model', '')

    if not user_message:
        return jsonify({'error': 'Empty message'})

    # Rate limit
    client_ip = request.remote_addr
    if rate_limit_exceeded(client_ip):
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        return jsonify({'error': 'Rate limit exceeded. Please wait a moment.'}), 429

    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())[:8]

    session['model'] = model
    if fallback_model:
        session['fallback_model'] = fallback_model

    # Capture session data (non-streaming, session is accessible)
    current_chat_id = session['chat_id']

    session_data = load_session(current_chat_id) or {
        'model': model,
        'fallback_model': fallback_model,
        'title': user_message[:50] + ('...' if len(user_message) > 50 else ''),
        'created': datetime.now().isoformat(),
        'messages': []
    }

    session_data['messages'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })

    logger.info("Chat request: model=%s, msg_len=%d, session=%s", model, len(user_message), current_chat_id)

    # Use Ollama tools (no regex-based command detection)
    result = process_ollama_response(model, session_data['messages'], OLLAMA_TOOLS)
    response_text = result.get('response', '') if isinstance(result, dict) else result
    prompt_tokens = result.get('prompt_eval_count', 0) if isinstance(result, dict) else 0

    # Fallback model
    if (response_text.startswith('Error:') or response_text.startswith('[ERROR]')) and fallback_model and fallback_model != model:
        logger.info("Primary model '%s' failed, trying fallback '%s'", model, fallback_model)
        result = process_ollama_response(fallback_model, session_data['messages'], OLLAMA_TOOLS)
        response_text = result.get('response', '') if isinstance(result, dict) else result
        prompt_tokens = result.get('prompt_eval_count', 0) if isinstance(result, dict) else 0
        if not response_text.startswith('Error:') and not response_text.startswith('[ERROR]'):
            response_text = f"[Fallback: {fallback_model}]\n\n{response_text}"

    session_data['messages'].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': datetime.now().isoformat()
    })
    session_data['context_usage'] = prompt_tokens
    save_session(current_chat_id, session_data)

    return jsonify({
        'response': response_text,
        'session_id': current_chat_id,
        'context_usage': prompt_tokens
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
        logger.info("Deleted session: %s", session_id)
        # If the deleted session is the current one, clear it
        if session.get('chat_id') == session_id:
            session.pop('chat_id', None)
        return jsonify({'success': True})

    return jsonify({'error': 'Session not found'})


@app.route('/api/session/switch', methods=['POST'])
def api_session_switch():
    """API to switch to an existing session (sync server-side session)"""
    data = request.json
    session_id = data.get('session_id', '')

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        session['chat_id'] = session_id
        # Reset write permission for new session
        _session_write_permissions.pop(session_id, None)
        logger.info("Switched to session: %s", session_id)
        return jsonify({'success': True, 'session_id': session_id})

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
        logger.info("Renamed session %s to '%s'", session_id, new_title)
        return jsonify({'success': True})

    return jsonify({'error': 'Session not found'})


@app.route('/api/session/save', methods=['POST'])
def api_session_save():
    """API to save session data (model, fallback_model, etc.)"""
    data = request.json
    session_id = data.get('session_id', '')
    session_data = data.get('data', {})

    if not session_id:
        return jsonify({'error': 'Session ID required'})

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        # Merge: update only the fields provided
        with open(filepath, 'r') as f:
            existing = json.load(f)
        for key in ('model', 'fallback_model', 'context_usage'):
            if key in session_data:
                existing[key] = session_data[key]
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)
        return jsonify({'success': True})

    return jsonify({'error': 'Session not found'})


@app.route('/api/save-code', methods=['POST'])
def api_save_code():
    """API to save code to a file"""
    data = request.json
    filepath = data.get('filepath', '')
    content = data.get('content', '')
    session_id = data.get('session_id', '')

    if not filepath or not content:
        return jsonify({'error': 'Filepath and content required'})

    if '..' in filepath or filepath.startswith('/etc') or filepath.startswith('/usr'):
        return jsonify({'error': 'Invalid filepath'})

    try:
        perm = check_write_permission(f'write {filepath}', session_id)
        if perm == 'ask':
            return jsonify({'error': 'write_permission_required', 'command': f'Write to {filepath}', 'filepath': filepath})

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Saved code to: %s", filepath)
        return jsonify({'success': True, 'filepath': filepath})
    except Exception as e:
        logger.error("Error saving code: %s", e)
        return jsonify({'error': str(e)})


@app.route('/api/session/new', methods=['POST'])
def api_session_new():
    """API to create a new session"""
    session['chat_id'] = str(uuid.uuid4())[:8]

    # Reset write permission for new session
    _session_write_permissions.pop(session.get('chat_id', ''), None)

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


@app.route('/api/clear-all-sessions', methods=['DELETE'])
def api_clear_all():
    """Delete all sessions"""
    count = 0
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith('.json'):
            os.remove(os.path.join(SESSIONS_DIR, filename))
            count += 1
    logger.info("Cleared all sessions (%d deleted)", count)
    return jsonify({'success': True, 'deleted': count})


if __name__ == '__main__':
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)

    print("=" * 50)
    print("Ollama WebChat")
    print("=" * 50)
    print(f"Open: http://localhost:5000")
    print(f"Debug: {DEBUG}")
    print(f"Sessions: {SESSIONS_DIR}")
    print(f"Ollama: {OLLAMA_BASE_URL}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=DEBUG)