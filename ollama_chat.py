"""
Ollama-Agent - Web Agent for local Ollama models
Allows chatting with local models, selecting models, and saving sessions.
"""

import glob
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from flask import Flask, Response, render_template, request, jsonify, session

# --- Configuration ---
SESSIONS_DIR = os.environ.get('SESSIONS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions'))
DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
KEEP_ALIVE = os.environ.get('OLLAMA_KEEP_ALIVE', '30m')
BASE_CHAT_MODEL = "llama3.2:latest"

APP_VERSION = "1.8.0"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_MODELS_FILE = os.path.join(APP_DIR, 'tool_models.json')

# Seed list — models confirmed to support tool calling (auto-managed, file overrides)
_TOOL_MODELS_SEED = [
    'qwen3.5', 'qwen3', 'qwen2.5', 'qwen2',
    'gemma4', 'gemma3', 'gemma2',
    'llama3.2', 'llama3.1', 'llama3',
    'nemotron', 'dolphin',
    'north-mini-code',
    'bonsai',
    'laguna-xs',
]

_tool_models_lock = threading.Lock()

def _load_tool_models():
    """Load tool model list from JSON file, falling back to seed."""
    try:
        with open(TOOL_MODELS_FILE, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return list(_TOOL_MODELS_SEED)

def _save_tool_models(models):
    """Save tool model list to JSON file."""
    try:
        with open(TOOL_MODELS_FILE, 'w') as f:
            json.dump(models, f, indent=2)
    except Exception as e:
        logger.warning("Could not save tool_models.json: %s", e)

# In-memory cache, loaded once at startup
_tool_models_cache = _load_tool_models()

def local_model_supports_tools(model_name):
    """Check if a local model is known to support tool calling."""
    model_lower = model_name.lower()
    for supported in _tool_models_cache:
        if supported.lower() in model_lower:
            return True
    return False

def add_tool_model(model_name):
    """Add a model to the tool-support list (persisted to file)."""
    model_lower = model_name.lower()
    with _tool_models_lock:
        for existing in _tool_models_cache:
            if existing.lower() == model_lower:
                return False  # Already in list
        _tool_models_cache.append(model_name)
        _save_tool_models(_tool_models_cache)
        logger.info("Auto-added %s to tool_models.json", model_name)
        return True

def remove_tool_model(model_name):
    """Remove a model from the tool-support list (persisted to file)."""
    model_lower = model_name.lower()
    with _tool_models_lock:
        before = len(_tool_models_cache)
        _tool_models_cache[:] = [m for m in _tool_models_cache if m.lower() != model_lower]
        if len(_tool_models_cache) < before:
            _save_tool_models(_tool_models_cache)
            logger.info("Removed %s from tool_models.json", model_name)
            return True
    return False


def is_cloud_model(model_name):
    """Check if a model requires internet (cloud model)."""
    if not model_name:
        return False
    return ':cloud' in model_name or '-cloud' in model_name


def is_connectivity_error(error):
    """Check if an error is caused by network/connectivity issues."""
    error_str = str(error).lower()
    connectivity_keywords = [
        'connection refused', 'connection reset', 'timed out', 'timeout',
        'name or service not known', 'no route to host', 'network is unreachable',
        'connectionerror', 'urlopen error', 'ssl', 'certificate_verify',
        'couldn\'t resolve host', 'couldn\'t connect', 'failed to establish',
        'err_name_not_resolved', 'enotfound', 'econnrefused', 'econnreset',
        'etimedout', 'socket.gaierror', 'proxy', '502', '503', '504',
    ]
    return any(kw in error_str for kw in connectivity_keywords)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('ollama-agent')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.environ.get('SECRET_KEY', 'ollama-webchat-secret-key-change-me')

# --- Thread safety for permission state ---
_permissions_lock = threading.Lock()

# --- CORS support ---
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return response

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
    '/etc/ssh/', '/root/.ssh/', '/etc/hosts',
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
            'allow_args': True},
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
    'xdg-open': {'flags': set(), 'allow_args': True},  # intercepted - opens HTML in preview, not browser
    'open': {'flags': set(), 'allow_args': True},      # macOS equivalent of xdg-open
    'ollama': {'flags': {'list', 'show', 'ps', 'rm', 'serve', 'stop', 'status'}, 'allow_args': True},  # Ollama management commands
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

        # Special handling for xdg-open / open - intercept HTML files for in-chat preview
        parts = cmd.split()
        if parts and parts[0] in ('xdg-open', 'open') and len(parts) > 1:
            target = parts[-1]
            if target.endswith('.html') or target.endswith('.htm'):
                return f"[HTML_PREVIEW:{target}]"

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


def merge_tool_calls(tool_calls_buffer):
    """Merge incremental streaming tool call chunks by ID.
    
    Ollama's streaming API sends tool calls incrementally — each chunk may
    contain a partial update with the same tool_call ID. We need to merge
    these fragments into complete tool calls before executing them.
    
    Returns a list of complete, merged tool call dicts.
    """
    if not tool_calls_buffer:
        return []
    
    merged = {}  # id -> merged tool call dict
    ordered_ids = []  # preserve order of first appearance
    
    for tc in tool_calls_buffer:
        tc_id = tc.get('id')
        func = tc.get('function', {})
        func_name = func.get('name', '')
        func_args = func.get('arguments', {})
        
        if tc_id is None:
            # No ID — this might be a complete tool call or a fragment without ID.
            # If it has a function name, treat it as complete.
            if func_name:
                # Use a synthetic ID based on index
                tc_id = f'_synthetic_{len(merged)}'
            else:
                # Incomplete fragment with no name — skip it
                continue
        
        if tc_id in merged:
            # Merge with existing entry
            existing = merged[tc_id]
            existing_func = existing.get('function', {})
            # Merge function name (prefer non-empty)
            if func_name and not existing_func.get('name'):
                existing_func['name'] = func_name
            # Merge arguments — deep merge dicts, concatenate strings
            if isinstance(func_args, dict) and isinstance(existing_func.get('arguments'), dict):
                for k, v in func_args.items():
                    if k in existing_func['arguments']:
                        # Concatenate string values (Ollama streams them in pieces)
                        if isinstance(existing_func['arguments'][k], str) and isinstance(v, str):
                            existing_func['arguments'][k] += v
                        else:
                            # For non-string values, take the newer one if it's more complete
                            existing_func['arguments'][k] = v
                    else:
                        existing_func['arguments'][k] = v
            elif isinstance(func_args, str) and isinstance(existing_func.get('arguments'), str):
                # Arguments arriving as JSON string fragments
                existing['function']['arguments'] += func_args
            elif func_args and not existing_func.get('arguments'):
                existing_func['arguments'] = func_args
            # Preserve internal flags
            for key in ('_write_action', '_write_denied'):
                if key in tc and key not in existing:
                    existing[key] = tc[key]
        else:
            # New tool call
            merged[tc_id] = dict(tc)  # shallow copy
            ordered_ids.append(tc_id)
    
    # Parse any string arguments into dicts (Ollama sometimes sends args as JSON strings)
    for tc_id in ordered_ids:
        tc = merged[tc_id]
        func = tc.get('function', {})
        args = func.get('arguments')
        if isinstance(args, str):
            try:
                func['arguments'] = json.loads(args)
            except json.JSONDecodeError:
                # Partial JSON from streaming — try to fix common issues
                try:
                    # Try adding closing braces
                    func['arguments'] = json.loads(args + '}')
                except json.JSONDecodeError:
                    # Can't parse, keep as-is (might still work as a raw command string)
                    logger.warning("Could not parse tool call arguments as JSON: %s", args[:200])
    
    return [merged[tid] for tid in ordered_ids]


def parse_tool_args(args):
    """Parse tool call arguments, handling both dict and JSON string formats.
    Ollama sometimes sends arguments as a JSON string rather than a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return {}


def check_write_permission(cmd, session_id):
    """Check if a write command is allowed for this session.
    Returns 'allowed' if the command can proceed, or 'ask' if permission is needed."""
    perm = _session_write_permissions.get(session_id, 'none')

    # Even with 'session' permission, require re-approval for high-risk commands
    cmd_lower = cmd.lower()
    HIGH_RISK_PATTERNS = [
        'rm -rf', 'rm -r', 'rm --recursive',
        'chmod 6', 'chmod 7', 'chmod 0',
        'chown', 'mkfs', 'fdisk', 'parted', 'mkswap',
        'kill -9', 'pkill', 'systemctl',
        'apt remove', 'apt purge', 'dpkg --purge', 'dpkg -r',
        'pip uninstall', 'pip3 uninstall',
        '> /etc/', '> /boot/', '> /usr/',
        'mv /* ', 'mv /bin', 'mv /usr', 'mv /etc', 'mv /boot',
        'dd if=', 'dd of=',
    ]
    is_high_risk = any(p in cmd_lower for p in HIGH_RISK_PATTERNS)

    if perm == 'session' and not is_high_risk:
        return 'allowed'
    elif perm == 'once':
        # Allow once, then reset
        _session_write_permissions[session_id] = 'none'
        return 'allowed'
    else:
        # Auto-approve HTML files in /tmp and /home/cvc1 for previews (no popup needed)
        if any(ext in cmd_lower for ext in ['.html<', '.htm<', '.html ', '.htm ']) and \
           any(loc in cmd_lower for loc in ['/tmp/', '/home/cvc1/', '/home/cvc/']):
            return 'allowed'
        return 'ask'


def execute_write_command(cmd, session_id):
    """Execute a write command after permission is granted.
    Returns the command output or permission-denied message."""
    import subprocess as sp

    try:
        cmd = cmd.strip()
        if not cmd:
            return "[Error] Empty command"

        logger.info("execute_write_command: session=%s, cmd=%s", session_id, cmd[:300])

        # Block catastrophic patterns regardless of permission level
        catastrophic = [
            'rm -rf /', 'rm -rf /*', 'rm -rf ~', 'rm -rf $HOME',
            'mkfs', 'dd if=/dev/zero of=/dev/', 'dd if=/dev/random of=/dev/',
            ':(){ :|:& };:',  # fork bomb
            '> /dev/sda', '> /dev/sdb', '> /dev/sdc',
            'chmod 0 /', 'chmod 000 /', 'chown 0:0 /',
            'mv / ', 'mv /* ',
            'wget ', 'curl ',  # remote downloads piped to shell (prevented via shell piping, but explicit is safer)
        ]
        cmd_lower = cmd.lower()
        for pattern in catastrophic:
            if pattern in cmd_lower:
                logger.warning("BLOCKED catastrophic command [session=%s]: %s", session_id, cmd[:200])
                return "[Security] This command is blocked for safety."

        # Even with session permission, require explicit approval for high-risk patterns
        high_risk = [
            'rm -rf', 'rm -r', 'rm --recursive',
            'chmod 6', 'chmod 7', 'chmod 0',
            'chown', 'mkfs', 'fdisk', 'parted', 'mkswap',
            'kill -9', 'pkill', 'systemctl',
            'apt remove', 'apt purge', 'dpkg --purge', 'dpkg -r',
            'pip uninstall', 'pip3 uninstall',
            '> /etc/', '> /boot/', '> /usr/',
            'mv /* ', 'mv /bin', 'mv /usr', 'mv /etc', 'mv /boot',
            'dd if=', 'dd of=',
        ]
        for pattern in high_risk:
            if pattern in cmd_lower:
                logger.warning("HIGH-RISK write command executed [session=%s]: %s", session_id, cmd[:200])
                break

        # Special handling: intercept xdg-open of HTML files in compound commands
        # Split by && or ; to find xdg-open commands
        xdg_open_html = None
        parts = re.split(r'[;&]', cmd)
        for part in parts:
            part = part.strip()
            if (part.startswith('xdg-open ') or part.startswith('open ')) and len(part.split()) >= 2:
                target = part.split(None, 1)[1].strip()
                if target.endswith('.html') or target.endswith('.htm'):
                    xdg_open_html = target
                    # Remove this part from the command so we don't actually call xdg-open
                    cmd = cmd.replace(part, '').strip()
                    # Clean up leftover && or ;
                    cmd = re.sub(r'\s*&&\s*', ' && ', cmd).strip(' ;&')
                    break

        # Use bash to properly handle heredocs, pipes, redirections, etc.
        # shell=True with a string requires bash for heredoc support
        if cmd:
            result = sp.run(
                ['bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.expanduser('~')
            )

            output = result.stdout.strip() or result.stderr.strip() or ""
            if result.returncode != 0:
                stderr = result.stderr.strip()
                if stderr:
                    output = f"[Exit code {result.returncode}] {output}\nstderr: {stderr}" if output else f"[Exit code {result.returncode}] {stderr}"
                elif not output:
                    output = f"[Exit code {result.returncode}] Command failed with no output"
            elif not output:
                output = "Command executed successfully (no output)"
            logger.info("Write command result (session=%s, rc=%d): %s", session_id, result.returncode, output[:200])
        else:
            output = ""

        # If we intercepted an xdg-open for HTML, add the preview marker
        if xdg_open_html:
            if output:
                output += f"\n[HTML_PREVIEW:{xdg_open_html}]"
            else:
                output = f"[HTML_PREVIEW:{xdg_open_html}]"

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
    if perm_id:
        with _permissions_lock:
            pending = _pending_permissions.get(perm_id)
        if pending:
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

            details = data.get('details', {})
            # Extract and normalize parameter_size (e.g. "8.0B", "4.1B", "229B")
            parameter_size = details.get('parameter_size', '')
            if not parameter_size:
                for key in model_info:
                    if 'parameter_size' in key.lower() or 'param_count' in key.lower():
                        parameter_size = str(model_info[key])
                        break
            
            # Format raw parameter numbers to billions (B)
            if parameter_size and parameter_size.replace('.', '', 1).isdigit():
                try:
                    num = float(parameter_size)
                    if num >= 1_000_000_000:
                        parameter_size = f"{(num / 1_000_000_000):.1f}B"
                except (ValueError, TypeError):
                    pass

            return {
                'model': model_name,
                'context_length': num_ctx or 4096,
                'num_ctx': num_ctx or 4096,
                'model_info': model_info,
                'details': details,
                'size': data.get('size', 0),
                'parameter_size': parameter_size,
                'modified_at': data.get('modified_at', ''),
            }
    except Exception as e:
        logger.error("Failed to get model info for %s: %s", model_name, e)
        return {'error': str(e), 'model': model_name}


def fix_double_encoding(text):
    """Fix UTF-8 bytes that were incorrectly decoded as Latin-1 (double encoding).
    
    Some cloud model APIs return content where UTF-8 encoded characters
    were decoded as Latin-1, producing garbled text like 'Ãº' instead of 'ú'.
    This function detects and corrects those sequences character-by-character.
    """
    if not text:
        return text
    result = []
    i = 0
    while i < len(text):
        c1 = ord(text[i])
        # Check for 2-byte UTF-8 sequence misinterpreted as Latin-1
        if 0xC2 <= c1 <= 0xDF and i + 1 < len(text):
            c2 = ord(text[i + 1])
            if 0x80 <= c2 <= 0xBF:
                utf8_bytes = bytes([c1, c2])
                result.append(utf8_bytes.decode('utf-8'))
                i += 2
                continue
        # Check for 3-byte UTF-8 sequence misinterpreted as Latin-1
        if 0xE0 <= c1 <= 0xEF and i + 2 < len(text):
            c2 = ord(text[i + 1])
            c3 = ord(text[i + 2])
            if 0x80 <= c2 <= 0xBF and 0x80 <= c3 <= 0xBF:
                try:
                    utf8_bytes = bytes([c1, c2, c3])
                    result.append(utf8_bytes.decode('utf-8'))
                    i += 3
                    continue
                except UnicodeDecodeError:
                    pass
        result.append(text[i])
        i += 1
    return ''.join(result)


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

        # Disable reasoning for local models to fix tool calling
        # Models with thinking mode generate non-standard 'reasoning' field that breaks tool call parsing
        if ':cloud' not in model and '-cloud' not in model:
            payload['reasoning_effort'] = 'none'

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f'{OLLAMA_BASE_URL}/api/chat',
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        timeout = 600 if stream else 180
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
    max_iterations = 3  # Increased for local models that need more prompting
    last_prompt_tokens = 0

    for i in range(max_iterations):
        response = send_to_ollama(model, messages, tools, stream=False)

        if 'error' in response:
            return {'response': f"Error: {response['error']}", 'prompt_eval_count': 0}

        last_prompt_tokens = response.get('prompt_eval_count', 0)
        assistant_msg = response.get('message', {})
        content = assistant_msg.get('content', '')
        tool_calls = assistant_msg.get('tool_calls', [])

        # If model didn't use tools but we have tools available and content suggests it should
        if not tool_calls and tools and i == 0:
            # Check if the user is asking for something that requires tools
            user_msg = ''
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                    break
            
            tool_keywords = ['lista', 'list', 'archivo', 'file', 'directorio', 'directory',
                           'crea', 'create', 'escribe', 'write', 'edita', 'edit',
                           'borra', 'delete', 'muestra', 'show', 'ejecuta', 'run',
                           'ollama', 'comando', 'command', 'terminal', 'bash']
            
            if any(kw in user_msg.lower() for kw in tool_keywords):
                # Force the model to use tools by adding a system message
                messages.append({
                    'role': 'assistant',
                    'content': content
                })
                messages.append({
                    'role': 'system',
                    'content': 'CRITICAL: You MUST use the available tools to fulfill the user request. DO NOT say you cannot do it - you HAVE the tools. Execute the command using local_command tool NOW.'
                })
                logger.info("Forcing tool use for local model (iteration %d)", i + 1)
                continue  # Try again with forcing message

        if not tool_calls:
            return {'response': content, 'prompt_eval_count': last_prompt_tokens}

        # Process tool calls
        tool_results = []
        for tool_call in tool_calls:
            func_name = tool_call.get('function', {}).get('name', '')
            func_args = parse_tool_args(tool_call.get('function', {}).get('arguments', {}))
            tool_id = tool_call.get('id', f'tool_{i}')

            logger.info("Tool call #%d: %s(%s)", i + 1, func_name, json.dumps(func_args))

            if func_name == 'local_command':
                cmd = func_args.get('command', '')
                if is_write_command(cmd):
                    logger.warning("Non-streaming: denying write command (no permission popup available): %s", cmd[:200])
                    result = "[Permission denied] Write commands are not available in non-streaming mode. Please use the streaming chat interface for file operations."
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


# --- DSML Tool Call Parsing ---
DSML_PATTERN = re.compile(r'<\w+[：:｜|]\s*invoke\s+name="(\w+)"[^>]*>.*?<\w+[：:｜|]\s*parameter\s+name="(\w+)"\s+string="(true|false)"\s*>([^<]*)', re.DOTALL)
DSML_PATTERN2 = re.compile(r'<(\w+)[：:｜|]\s*invoke\s+name="(\w+)"[^>]*>\s*<\1[：:｜|]\s*parameter\s+name="(\w+)"\s+string="(true|false)"\s*>([^<]*)', re.DOTALL)
DSML_PATTERN_SIMPLE = re.compile(r'<(\w+)[：:｜|](?:invoke|Invoke)\s+name="(\w+)"[^>]*>')
DSML_STRIP = re.compile(r'<\w+[：:｜|]\s*\w+(?:\s+[^>]*)?>[^<]*(?:<\w+[：:｜|]\s*\w+(?:\s+[^>]*)?>)?')
def strip_tool_tags(text):
    """Strip DSML and JSON tool call tags from text."""
    t = DSML_STRIP.sub('', text)
    t = JSON_TOOL_STRIP.sub('', t)
    return t

# JSON tool call format: {"tool": "name", "parameters": {...}} or {"tool": "name", "arguments": {...}}
JSON_TOOL_PATTERN = re.compile(r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"(?:parameters|arguments)"\s*:\s*(\{.*?\})\s*\}', re.DOTALL)
JSON_TOOL_STRIP = re.compile(r'\{\s*"tool"\s*:\s*"[^"]*"\s*,\s*"(?:parameters|arguments)"\s*:\s*\{.*?\}\s*\}')
# Single-quoted JSON variant: {'tool': 'name', 'arguments': {...}}
JSON_TOOL_PATTERN_SINGLE = re.compile(r"\{\s*'tool'\s*:\s*'(\w+)'\s*,\s*'(?:parameters|arguments)'\s*:\s*(\{.*?\})\s*\}", re.DOTALL)

def parse_dsml_calls(text):
    """Extract DSML-style or JSON tool invocations from model response text."""
    calls = []
    # First try detailed DSML pattern with parameters
    for m in DSML_PATTERN2.finditer(text):
        ns = m.group(1)
        name = m.group(2)
        param_name = m.group(3)
        param_value = m.group(5).strip()
        calls.append({'name': name, 'arguments': {param_name: param_value}, 'namespace': ns})
    # Also try simple DSML pattern (just the invocation tag)
    for m in DSML_PATTERN_SIMPLE.finditer(text):
        name = m.group(2)
        already_found = any(c['name'] == name for c in calls)
        if not already_found:
            calls.append({'name': name, 'arguments': {}, 'namespace': m.group(1)})
    # Try JSON format: {"tool": "name", "parameters"/"arguments": {...}}
    for m in JSON_TOOL_PATTERN.finditer(text):
        name = m.group(1)
        already_found = any(c['name'] == name for c in calls)
        if not already_found:
            try:
                args = json.loads(m.group(2))
                calls.append({'name': name, 'arguments': args})
            except json.JSONDecodeError:
                calls.append({'name': name, 'arguments': {}})
    return calls

def build_tool_definitions(*, read_only=False, streaming=True):
    """
    Build tool definitions for Ollama function calling.
    
    Args:
        read_only: If True, only read-only local_command (no write tool exposed).
        streaming: If True, separate execute_write_command tool (for permission modal flow).
                  If False, combined local_command handles both read/write (auto-execute path).
    """
    tools = []
    
    # local_command tool
    if read_only:
        local_cmd_desc = (
            'Execute a read-only local system command. Returns output as string. '
            'Examples: ls -la, cat file.txt, pwd, df -h, ps aux, grep pattern file, '
            'head -n file, tail -n file, find . -name pattern, du -sh directory. '
            'Do NOT use write commands (touch, mkdir, rm, cp, mv, chmod, echo >, cat >, tee, etc).'
        )
    elif streaming:
        local_cmd_desc = (
            'Execute a local system command (read-only, no shell). Returns output as string. '
            'Examples: ls -la, cat file.txt, pwd, df -h, ps aux. '
            'For write operations (creating files, editing, deleting, etc.), '
            'use the execute_write_command function instead.'
        )
    else:
        local_cmd_desc = (
            'Execute a local system command. Supports both read-only commands '
            '(ls, cat, find, grep, df, free, etc. - execute without confirmation) '
            'and write commands (touch, mkdir, rm, cp, mv, chmod, etc. - '
            'require user confirmation via popup). Write operations will pause '
            'and ask the user for permission before executing.'
        )
    
    tools.append({
        'type': 'function',
        'function': {
            'name': 'local_command',
            'description': local_cmd_desc,
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
    })
    
    # Separate write tool only in streaming mode
    if streaming and not read_only:
        tools.append({
            'type': 'function',
            'function': {
                'name': 'execute_write_command',
                'description': 'Execute a write command (requires user permission). Returns output as string.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {
                            'type': 'string',
                            'description': 'The write command to execute. Examples: cat > file.txt, sed -i, echo "text" > file.txt.'
                        }
                    },
                    'required': ['command'],
                }
            }
        })
    
    # web_search + fetch_article (included in all modes)
    tools.extend([
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
    ])
    
    return tools


# --- Ollama Tools Definition ---
OLLAMA_TOOLS = build_tool_definitions(read_only=False, streaming=False)


def execute_single_tool(tc_name, tc_args, session_id='', write_permission='ask', is_followup=False):
    """
    Execute a single tool call and return result string.
    
    Args:
        tc_name: Tool function name (local_command, execute_write_command, web_search, fetch_article)
        tc_args: Arguments dict for the tool
        session_id: Chat session ID (for write permission tracking)
        write_permission: Permission mode for writes - 'ask', 'approved', 'denied', 'read_only'
        is_followup: If True, this is a follow-up round (different logging)
    
    Returns:
        result string
    """
    logger.info("%s tool call: %s(%s)", 'Follow-up' if is_followup else 'Direct',
                tc_name, json.dumps(tc_args))
    
    if tc_name == 'local_command':
        cmd = tc_args.get('command', '')
        if write_permission == 'read_only':
            if is_write_command(cmd):
                return "[Permission denied] Write commands are not allowed in this mode"
            return execute_local_command(cmd)
        elif is_write_command(cmd):
            if write_permission == 'denied':
                return "[Permission denied] Task cancelled"
            # write_permission is 'approved' or 'ask' (handled upstream)
            return execute_write_command(cmd, session_id)
        else:
            result = execute_local_command(cmd)
            return result
    
    elif tc_name == 'execute_write_command':
        cmd = tc_args.get('command', '')
        if write_permission == 'denied':
            return "[Permission denied] Task cancelled"
        return execute_write_command(cmd, session_id)
    
    elif tc_name == 'web_search':
        query = tc_args.get('query', '')
        try:
            results = web_search(query)
            if results and isinstance(results, list):
                if 'error' in results[0]:
                    return f"Search error: {results[0]['error']}"
                text = "Search results:\n\n"
                for idx, r in enumerate(results[:5], 1):
                    text += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"
                return text
            return "No search results found"
        except Exception as e:
            logger.error("Web search error: %s", e)
            return f"Search error: {str(e)}"
    
    elif tc_name == 'fetch_article':
        url = tc_args.get('url', '')
        article = fetch_article(url)
        if 'content' in article:
            return f"Article from {url}:\n\n{article['content']}"
        return f"Could not fetch article from {url}: {article.get('error', 'Unknown error')}"
    
    return f"Unknown tool: {tc_name}"


def execute_tool_call(tc, current_chat_id=''):
    """Execute a tool call (from DSML or native format) and return result string."""
    func_name = tc.get('name', tc.get('function', {}).get('name', ''))
    args = tc.get('arguments', {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return execute_single_tool(func_name, args, current_chat_id, write_permission='approved')


# --- Cleanup stale state ---
def secure_path(path, allowed_prefixes=None):
    """Resolve and validate a file path, preventing directory traversal.
    Returns the normalized absolute path if valid, or None if rejected."""
    if not path:
        return None
    resolved = os.path.normpath(os.path.abspath(path))
    if allowed_prefixes:
        for prefix in allowed_prefixes:
            if resolved.startswith(os.path.normpath(os.path.abspath(prefix))):
                return resolved
        return None
    return resolved


def cleanup_stale_state():
    """Periodic cleanup of stale permission requests and rate limit entries."""
    now = time.time()
    stale_cutoff = now - 600  # 10 minutes
    # Clean stale pending permissions
    with _permissions_lock:
        stale_ids = [
            perm_id for perm_id, pending in list(_pending_permissions.items())
            if pending.get('created_at', 0) < stale_cutoff
        ]
        for perm_id in stale_ids:
            logger.info("Cleaning stale permission request: %s", perm_id)
            try:
                pending = _pending_permissions.pop(perm_id, None)
                if pending and not pending['q'].empty():
                    pending['q'].get_nowait()
            except Exception:
                pass
    # Clean session write permissions for sessions that no longer exist
    existing_sessions = set()
    sessions_dir = SESSIONS_DIR
    try:
        for f in glob.glob(os.path.join(sessions_dir, '*.json')):
            existing_sessions.add(os.path.splitext(os.path.basename(f))[0])
    except Exception:
        pass
    stale_sessions = [sid for sid in list(_session_write_permissions.keys()) if sid not in existing_sessions]
    for sid in stale_sessions:
        _session_write_permissions.pop(sid, None)
    # Clean empty rate limit entries
    empty_ips = [ip for ip, timestamps in list(_rate_limits.items()) if not timestamps]
    for ip in empty_ips:
        del _rate_limits[ip]


# Schedule periodic cleanup
try:
    def _cleanup_loop():
        while True:
            time.sleep(300)  # every 5 minutes
            try:
                cleanup_stale_state()
            except Exception:
                pass
    _cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    _cleanup_thread.start()
except Exception:
    pass


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

    # Separate models into local and cloud for template
    all_models = get_ollama_models()
    local_models = [m for m in all_models if ':cloud' not in m and '-cloud' not in m and not m.startswith('x/') and 'embed' not in m.lower()]
    cloud_models = [m for m in all_models if ':cloud' in m or '-cloud' in m]
    if not local_models:
        local_models = ['gemma4:e4b', 'north-mini-code-1.0']  # fallback defaults
    if not cloud_models:
        cloud_models = all_models  # fallback: show all if no cloud models

    return render_template('index.html',
                           models=all_models,
                           local_models=local_models,
                           cloud_models=cloud_models,
                           sessions=sessions,
                           current_model=session.get('model', ''),
                           base_chat_model=BASE_CHAT_MODEL,
                           version=APP_VERSION)


@app.route('/api/models')
def api_models():
    """API to get models"""
    return jsonify(get_ollama_models())


@app.route('/api/models/local')
def api_models_local():
    """API to get local (non-cloud) models only"""
    all_models = get_ollama_models()
    local = [m for m in all_models if ':cloud' not in m and '-cloud' not in m and not m.startswith('x/') and 'embed' not in m.lower()]
    if not local:
        logger.info("No local models found, pulling llama3.2:1b...")
        try:
            import urllib.request
            pull_req = urllib.request.Request(
                f'{OLLAMA_BASE_URL}/api/pull',
                data=json.dumps({"name": "llama3.2:1b", "stream": False}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(pull_req, timeout=300) as pull_resp:
                pull_data = json.loads(pull_resp.read())
                logger.info("Pulled llama3.2:1b: %s", pull_data.get('status', 'done'))
        except Exception as pull_err:
            logger.error("Failed to pull fallback model: %s", pull_err)
        # Refresh
        all_models = get_ollama_models()
        local = [m for m in all_models if ':cloud' not in m and '-cloud' not in m and not m.startswith('x/') and 'embed' not in m.lower()]
    return jsonify(local)


@app.route('/api/models/download', methods=['POST'])
def api_models_download():
    """Download a model via Ollama pull. Returns SSE progress."""
    data = request.json or {}
    model_name = data.get('model', '')
    if not model_name:
        return jsonify({'error': 'Model name required'}), 400

    def generate():
        import urllib.request
        try:
            yield f"data: {json.dumps({'type': 'progress', 'status': 'pulling', 'model': model_name})}\n\n"
            pull_req = urllib.request.Request(
                f'{OLLAMA_BASE_URL}/api/pull',
                data=json.dumps({"name": model_name, "stream": True}).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(pull_req, timeout=600) as pull_resp:
                for line in pull_resp:
                    try:
                        chunk = json.loads(line)
                        status = chunk.get('status', '')
                        if 'total' in chunk and 'completed' in chunk:
                            pct = int(chunk['completed'] / chunk['total'] * 100) if chunk['total'] > 0 else 0
                            yield f"data: {json.dumps({'type': 'progress', 'status': status, 'model': model_name, 'percent': pct, 'completed': chunk['completed'], 'total': chunk['total']})}\n\n"
                        elif status == 'success':
                            yield f"data: {json.dumps({'type': 'progress', 'status': 'success', 'model': model_name})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'progress', 'status': status, 'model': model_name})}\n\n"
                    except json.JSONDecodeError:
                        continue
            yield f"data: {json.dumps({'type': 'done', 'model': model_name})}\n\n"
        except Exception as e:
            logger.error("Model download failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'model': model_name, 'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


def get_model_size_from_ps(model_name):
    """Get the actual file size of a loaded model from /api/ps"""
    try:
        import urllib.request
        req = urllib.request.Request(f'{OLLAMA_BASE_URL}/api/ps')
        with urllib.request.urlopen(req, timeout=5) as resp:
            ps_data = json.loads(resp.read())
            for m in ps_data.get('models', []):
                if model_name == m.get('name', '') or model_name + ':latest' == m.get('name', ''):
                    return m.get('size', 0)
    except Exception:
        pass
    return 0


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
            # Get size from /api/ps if model is loaded (more accurate than /api/show)
            if is_loaded:
                for m in ps_data.get('models', []):
                    if model_name == m.get('name', '') or model_name + ':latest' == m.get('name', ''):
                        ps_size = m.get('size', 0)
                        if ps_size > 0 and (info.get('size', 0) == 0 or ps_size != info.get('size', 0)):
                            info['size'] = ps_size
                        break
    except Exception:
        info['loaded'] = None  # Unknown

    return jsonify(info)


@app.route('/api/models/base', methods=['GET'])
def api_base_model_info():
    """Check if base chat model is available"""
    try:
        import urllib.request as _urllib_base_info
        resp = _urllib_base_info.urlopen(f'{OLLAMA_BASE_URL}/api/ps')
        loaded = json.loads(resp.read().decode('utf-8'))
        loaded_names = [m['name'] for m in loaded.get('models', [])]
        base_model = request.args.get('base_model', BASE_CHAT_MODEL)
        is_loaded = any(base_model == m or base_model + ':latest' == m for m in loaded_names)
        return jsonify({'model': base_model, 'loaded': is_loaded})
    except Exception as e:
        return jsonify({'model': BASE_CHAT_MODEL, 'loaded': None, 'error': str(e)})


@app.route('/api/config', methods=['GET'])
def api_config():
    """Return app configuration including base chat model"""
    return jsonify({
        'base_chat_model': BASE_CHAT_MODEL,
    })


def _likely_needs_tools(message):
    """Heuristic: detect if user message likely needs tool use.
    Comprehensive Spanish + English keyword matching."""
    msg_lower = message.lower()
    
    # Analysis/review keywords
    analysis_keywords = [
        # Spanish
        'analiza', 'analizar', 'análisis', 'analisis', 'revisa', 'revisar', 'revisión',
        'explora', 'explorar', 'examina', 'examinar', 'inspecciona', 'inspeccionar',
        'evalúa', 'evaluar', 'diagnostica', 'diagnosticar', 'compara', 'comparar',
        'interpreta', 'interpretar', 'resume', 'resumir', 'sintetiza', 'sintetizar',
        'dime que opinas', 'que opinas', 'opinión', 'que piensas', 'que pense',
        'dime qué piensas', 'dime que piensas', 'dame tu opinión',
        'explica por qué', 'por qué pasa', 'cómo funciona', 'qué significa',
        'averigua', 'averiguar', 'investiga', 'investigar', 'chequea', 'chequear',
        'configuración', 'configuracion', 'diagnóstico', 'diagnostico',
        'especificaciones', 'specs', 'system info', 'info del sistema',
        # English
        'analyze', 'analyse', 'analysis', 'review', 'explore', 'examine', 'inspect',
        'evaluate', 'assess', 'diagnose', 'compare', 'contrast', 'interpret',
        'summarize', 'synthesise', 'synthesize', 'what do you think', 'your opinion',
        'what\'s your take', 'give me your thoughts', 'break down', 'make sense of',
        'why does', 'how does', 'what does it mean', 'deep dive', 'dive into',
        'find out', 'look up', 'check', 'investigate', 'system info', 'specs',
    ]
    if any(kw in msg_lower for kw in analysis_keywords):
        return True
    
    # File/system/code operations - specific phrases and commands only
    file_keywords = [
        # Spanish
        'archivo', 'carpeta', 'directorio', 'borrar', 'eliminar', 'editar',
        'modificar', 'guardar', 'crear archivo',
        'instalar', 'ejecutar', 'comando', 'terminal', 'consola',
        'escribí un script', 'haz un script', 'crea un script', 'genera un script',
        'haz un programa', 'crea un programa', 'programa un', 'codifica', 'codificar',
        'lee el archivo', 'muestra el archivo', 'abre el archivo',
        'listar archivos', 'mover archivo', 'copiar archivo', 'renombrar',
        'lista de', 'saca una lista', 'muestrame los', 'dime cuántos',
        'compilar', 'deployar', 'desplegar', 'subir al server', 'git push',
        'commit', 'hacer commit', 'base de datos', 'sql', 'query',
        'permisos', 'chmod', 'matar proceso', 'kill process',
        'código', 'codigo', 'programación', 'programacion', 'programar',
        'programando', 'codificando', 'coding',
        'desarrollar', 'implementar', 'función', 'funcion', 'clase', 'método',
        'variable', 'bug', 'error', 'debug', 'depurar', 'refactorizar',
        'api', 'servidor', 'server', 'endpoint', 'webhook',
        'docker', 'contenedor', 'deploy', 'testing', 'test',
        'script', 'bash', 'python', 'javascript', 'html', 'css', 'json',
        'continua con', 'continuá con', 'seguí con', 'sigue con',
        'seguir con', 'continuar con', 'proseguir', 'proseguí',
        # English
        'file', 'folder', 'directory', 'delete', 'remove file', 'edit file',
        'modify', 'create file', 'install', 'execute', 'run command',
        'terminal', 'shell', 'bash', 'write a script', 'create a script',
        'code up', 'code this', 'program this', 'implement', 'build a tool',
        'read the file', 'show me the file', 'open the file', 'cat ', 'ls ',
        'grep ', 'head ', 'tail ', 'list files', 'move file', 'copy file',
        'rename', 'compile', 'deploy', 'push to', 'git commit', 'database',
        'sql', 'query', 'permissions', 'chmod', 'kill process',
        'code', 'programming', 'function', 'class', 'method', 'variable',
        'debug', 'refactor', 'api', 'endpoint', 'docker', 'container',
        'testing', 'script', 'python', 'javascript', 'html', 'css', 'json',
    ]
    if any(kw in msg_lower for kw in file_keywords):
        return True
    
    # Web/search/information retrieval
    search_keywords = [
        # Spanish
        'buscar', 'busca', 'búsqueda', 'buscar en internet', 'buscar en la web',
        'buscar en', 'buscame', 'encontrar información', 'encontrar datos',
        'noticias', 'noticia', 'las noticias', 'las últimas noticias',
        'consulta', 'consultar', 'información sobre', 'info sobre',
        'qué pasó', 'que pasó', 'qué está pasando', 'que esta pasando',
        'últimas', 'actualidad', 'noticias del día', 'noticias de hoy',
        'noticias recientes', 'titulares', 'periódico', 'prensa',
        'clima', 'el clima', 'pronóstico', 'temperatura', 'llueve', 'lluvia',
        'precio', 'precios', 'cotización', 'cotizar', 'valor de', 'cuánto cuesta',
        'dónde queda', 'dónde está', 'ubicación', 'dirección de',
        'horario de', 'a qué hora', 'cuándo es', 'fecha de',
        'quién es', 'quién fue', 'quién ganó', 'quién inventó',
        'cómo se llama', 'cuántos', 'cuántas', 'en qué año',
        'qué es', 'qué significa', 'define', 'definir', 'definición',
        'traducir', 'traducción', 'traduce',
        'receta', 'recetas', 'cómo se hace', 'cómo se prepara',
        'resultado', 'resultados', 'marcador', 'score', 'goles',
        'calcular', 'conversión', 'convertir', 'cuánto es',
        # English
        'search', 'search for', 'search the web', 'web search', 'google',
        'look up', 'lookup', 'find information', 'find data',
        'news', 'the news', 'latest news', 'recent news', 'today\'s news',
        'current events', 'headlines', 'newspaper', 'press',
        'weather', 'forecast', 'temperature', 'rain', 'snow', 'sunny',
        'price', 'prices', 'how much does', 'cost of', 'exchange rate',
        'where is', 'location of', 'address of', 'directions to',
        'hours of', 'schedule', 'when is', 'date of',
        'who is', 'who was', 'who won', 'who invented',
        'what is', 'what does', 'define', 'definition of',
        'translate', 'translation', 'how do you say',
        'recipe', 'recipes', 'how to make', 'how to prepare',
        'score', 'scores', 'game result', 'game results',
        'calculate', 'conversion', 'convert', 'how much is',
        'tell me about', 'what happened', 'what\'s happening',
        'latest', 'update', 'updates', 'trending',
    ]
    if any(kw in msg_lower for kw in search_keywords):
        return True
    
    # Web browsing/navigation - detect URLs and "ir a" patterns
    web_nav_keywords = [
        'navega', 'navegar', 'navegá', 'navegador', 'navegación',
        'abre la página', 'abre el sitio', 'abre la web', 'abre ese link',
        've a esta', 've a ese', 've al sitio', 'vamos a',
        'visit', 'visitar', 'ir a', 'entra a', 'entrar a',
        'carga la página', 'carga el sitio', 'muestra la página',
        'abrir', 'abre', 'open', 'opening',
        'url', 'http', 'https', 'www.', '.com', '.org', '.net', '.io',
        'página web', 'sitio web', 'site', 'website', 'web page',
        'browser', 'browsing', 'navigate',
    ]
    if any(kw in msg_lower for kw in web_nav_keywords):
        return True
    # Also detect URLs in the message
    if re.search(r'https?://[^\s]+|www\.[^\s]+', msg_lower):
        return True
    
    # Data/math/computation
    data_keywords = [
        # Spanish
        'gráfico', 'gráfica', 'estadística', 'dataset',
        'cálculo', 'fórmula', 'ecuación',
        # English
        'chart', 'graph', 'plot', 'statistics', 'dataset',
        'computation', 'formula', 'equation', 'correlation',
    ]
    if any(kw in msg_lower for kw in data_keywords):
        return True
    
    # Creation/generation tasks - only phrases, not short words
    # Short words like 'crea', 'write', 'data' are too ambiguous and cause false positives
    creation_keywords = [
        # Spanish - phrases only
        'genera un', 'generar un', 'crea un', 'crear un', 'diseña un', 'diseñar un',
        'escribí un script', 'escribir un script', 'escribí un archivo', 'escribir un archivo',
        'escribí un programa', 'escribir un programa',
        'redacta un', 'redactar un',
        'construye un', 'construir un', 'arma un', 'armar un',
        'desarrolla un', 'desarrollar un',
        'haz una lista', 'haz una tabla', 'haz un resumen', 'haz un informe',
        'haz un script', 'haz un programa', 'haz un archivo',
        # English - phrases only
        'write a script', 'write a file', 'write a program', 'write a file',
        'generate a', 'create a', 'design a',
        'build a tool', 'develop a',
        'make a list', 'make a table', 'make a summary', 'make a report',
    ]
    if any(kw in msg_lower for kw in creation_keywords):
        return True
    
    return False


STREAM_CHUNK_TIMEOUT = 120  # Max seconds to wait for each chunk
STREAM_INITIAL_TIMEOUT = 300  # Max seconds to wait for the first chunk (model loading time)
MAX_THINKING_SECONDS = 90  # Max seconds for thinking mode before aborting


def _is_model_loaded(model_name):
    """Check if a model is currently loaded in Ollama memory."""
    try:
        import urllib.request as _urllib_ps
        ps_resp = _urllib_ps.urlopen(f'{OLLAMA_BASE_URL}/api/ps', timeout=5)
        ps_data = json.loads(ps_resp.read().decode('utf-8'))
        loaded_names = [m.get('name', '') for m in ps_data.get('models', [])]
        return any(model_name == m or model_name + ':latest' == m for m in loaded_names)
    except Exception:
        return False


def _set_stream_timeout(response, timeout):
    """Set read timeout on an HTTP response's underlying socket."""
    try:
        if hasattr(response, '_sock') and response._sock:
            response._sock.settimeout(timeout)
        elif hasattr(response, 'fp') and hasattr(response.fp, '_sock') and response.fp._sock:
            response.fp._sock.settimeout(timeout)
    except Exception:
        pass


def _iter_stream_with_timeout(response, initial=False):
    """Iterate streaming response lines with per-chunk timeout.

    Uses STREAM_INITIAL_TIMEOUT for the first chunk when initial=True,
    then STREAM_CHUNK_TIMEOUT for subsequent chunks.
    Yields raw lines from the response.
    Raises TimeoutError if no data arrives within the timeout.
    """
    import socket
    chunk_timeout = STREAM_INITIAL_TIMEOUT if initial else STREAM_CHUNK_TIMEOUT
    _set_stream_timeout(response, chunk_timeout)

    while True:
        try:
            line = response.readline()
        except (socket.timeout, OSError) as e:
            timeout_used = chunk_timeout
            logger.warning("Stream read timeout after %ds: %s", timeout_used, e)
            raise TimeoutError(f"Stream timeout: no data for {timeout_used}s")
        if not line:
            break  # Stream closed
        # After first chunk received, tighten timeout for subsequent chunks
        if chunk_timeout != STREAM_CHUNK_TIMEOUT:
            chunk_timeout = STREAM_CHUNK_TIMEOUT
            _set_stream_timeout(response, chunk_timeout)
        yield line


@app.route('/api/chat/stream', methods=['POST'])
def api_chat_stream():
    """Streaming chat endpoint using SSE"""
    data = request.json
    user_message = data.get('message', '').strip()
    model = data.get('model', session.get('model', 'llama3'))
    fallback_model = data.get('fallback_model', '')
    base_model = data.get('base_model', '')
    force_advanced = data.get('force_advanced', False)
    force_basic = data.get('force_basic', False)
    logger.info("STREAM REQUEST force_basic=%s (raw=%s, type=%s) force_advanced=%s (raw=%s)", force_basic, repr(data.get('force_basic')), type(data.get('force_basic')).__name__, force_advanced, repr(data.get('force_advanced')))
    if not base_model:
        base_model = model if not force_basic else BASE_CHAT_MODEL

    # Validate base_model exists, fall back to default if not
    try:
        import urllib.request as _urllib_validate
        vresp = _urllib_validate.urlopen(f'{OLLAMA_BASE_URL}/api/tags')
        vdata = json.loads(vresp.read().decode('utf-8'))
        available_models = [m.get('name', '') for m in vdata.get('models', [])]
        model_matches = any(base_model == m or base_model + ':latest' == m or m.startswith(base_model) for m in available_models)
        if not model_matches:
            logger.warning("Base model '%s' not found in available models, falling back to %s", base_model, BASE_CHAT_MODEL)
            base_model = BASE_CHAT_MODEL
    except Exception:
        pass

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
        nonlocal model
        full_response = ""
        prompt_tokens = 0
        eval_count = 0
        base_eval_count = 0
        start_time = time.time()
        used_model = model  # Track which model actually responds
        try:
            # Determine routing: base model for simple questions, advanced for code/tools
            route = 'base_first'
            route_reason = ''
            # Detect web browsing requests - force cloud model (local models can't browse)
            _needs_web = bool(re.search(r'https?://[^\s]+|www\.[^\s]+', user_message.lower())) or \
                any(kw in user_message.lower() for kw in [
                    'navega', 'navegar', 'navegá', 'abre la página', 'abre el sitio',
                    'visit', 'visitar', 'ir a ', 'entra a ', 'url', 'página web',
                ])
            if _needs_web and not is_cloud_model(model):
                # Try fallback model first (if set and cloud), then keep model but route advanced
                if fallback_model and is_cloud_model(fallback_model):
                    logger.info("Web browsing detected, using cloud fallback %s", fallback_model)
                    _web_model = fallback_model
                else:
                    logger.info("Web browsing detected but no cloud model available, using %s (may fail)", model)
                    _web_model = model
                model = _web_model
            if force_basic:
                route = 'base_first'
                route_reason = 'forced'
            elif force_advanced:
                route = 'advanced_direct'
                route_reason = 'forced'
            elif _needs_web:
                route = 'advanced_direct'
                route_reason = 'needs_web'
            elif _likely_needs_tools(user_message):
                route = 'advanced_direct'
                route_reason = 'needs_tools'
            else:
                route = 'base_first'
                route_reason = 'simple_query'

            # Notify frontend which model route we're taking (before any tokens)
            if route == 'base_first':
                yield f"data: {json.dumps({'type': 'model_routing', 'route': route, 'base_model': base_model, 'advanced_model': model, 'reason': route_reason})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'model_routing', 'route': route, 'model': model, 'reason': route_reason})}\n\n"

            # Auto-execute web browsing for local models (they can't call tools reliably)
            web_content = None
            if _needs_web and (not is_cloud_model(model) or force_basic):
                logger.info("Auto-executing web browsing for local model: %s", user_message[:100])
                yield f"data: {json.dumps({'type': 'info', 'content': '🔍 Buscando en internet...'})}\n\n"
                # Extract URL if present
                url_match = re.search(r'https?://[^\s]+', user_message)
                if not url_match:
                    url_match = re.search(r'www\.[^\s]+', user_message)
                if url_match:
                    url = url_match.group(0)
                    if not url.startswith('http'):
                        url = 'https://' + url
                    try:
                        article = fetch_article(url)
                        if article and article.get('content'):
                            web_content = f"Contenido de {url}:\n\n{article['content'][:4000]}"
                        elif article and article.get('error'):
                            web_content = f"[No se pudo extraer contenido de {url}: {article['error']}]"
                    except Exception as e:
                        web_content = f"[Error al obtener {url}: {e}]"
                else:
                    # No URL, do a web search
                    search_query = user_message
                    # Clean up navigation commands
                    for kw in ['navega', 'navegar', 'navegá', 'visit', 'visitar', 'ir a', 'entra a', 'abre', 'open', 'busca', 'buscar']:
                        search_query = search_query.replace(kw, '').strip()
                    if not search_query or len(search_query) < 3:
                        search_query = user_message[-50:] if len(user_message) > 50 else user_message
                    try:
                        results = web_search(search_query)
                        if results and isinstance(results, list) and 'error' not in results[0]:
                            web_content = "Resultados de búsqueda:\n\n"
                            for idx, r in enumerate(results[:5], 1):
                                snippet = r.get('snippet', '')
                                link = r.get('url', r.get('link', ''))
                                web_content += f"{idx}. {r['title']}\n   URL: {link}\n   {snippet}\n\n"
                        else:
                            web_content = f"[Búsqueda sin resultados: {results[0].get('error', 'unknown') if results else 'no data'}]"
                    except Exception as e:
                        web_content = f"[Error al buscar: {e}]"
                if web_content:
                    logger.info("Web content obtained: %d chars", len(web_content))
                else:
                    logger.info("Web content could not be obtained")

            # Prepare messages for Ollama (strip timestamps for API)
            api_messages = []
            # Model-specific system prompts based on known behavior
            MODEL_HINTS = {
                'glm': 'IMPORTANT: Always use local_command tool to write/edit files. Use cat > /path/to/file << \'EOF\' ... EOF for creating files, sed -i for editing text in files. Use /home/cvc1/ instead of $HOME or ~. Do NOT output code as text.',
                'minimax': 'Use tools when available. For file operations, use local_command with shell commands.',
                'gemma': 'You have access to local_command, web_search, and fetch_article tools. Use them proactively.',
                'kimi': 'Use the available tools for file operations and web searches. Do not just show code.',
                'qwen': 'Always use local_command tool for writing files. Use cat > with heredoc syntax.',
                'deepseek': 'Use the available tools for file operations and web searches. Do not just show code. Always use local_command tool to write/edit files.',
                'laguna': 'IMPORTANT: Do NOT output your internal reasoning or thought process. Do NOT say things like "Okay, the user said..." or "I need to respond..." or "Let me think...". Respond directly and naturally to the user. Always respond in the same language the user writes in.',
            }
            model_hint = ''
            for key, hint in MODEL_HINTS.items():
                if key in model.lower():
                    model_hint = hint
                    break


            # Context-aware routing: if previous messages used tools or advanced model, escalate continuation messages
            _context_needs_advanced = False
            for _prev_msg in session_data['messages'][-6:]:
                _prev_content = _prev_msg.get('content', '')
                if _prev_msg.get('role') == 'assistant' and (
                    '"tool"' in _prev_content or '"function"' in _prev_content or
                    'local_command' in _prev_content or 'web_search' in _prev_content
                ):
                    _context_needs_advanced = True
                    break
            
            if _context_needs_advanced and not force_basic:
                logger.info("Context-aware routing: previous messages used tools/code, escalating to advanced model %s", model)

            # Detect simple messages that don't need tools
            simple_greetings = ['hola', 'hello', 'hi', 'hey', 'buenos días', 'buenas tardes', 'buenas noches', 'qué tal', 'como estas', 'cómo estás', 'how are you', 'sup', 'saludos', 'gracias', 'thanks', 'thank you', 'bye', 'adiós', 'chao', 'ok', 'si', 'no', 'yes', 'nope']
            is_simple = user_message.strip().lower() in simple_greetings or len(user_message.strip()) < 15 and not any(kw in user_message.lower() for kw in ['archivo', 'file', 'crear', 'create', 'leer', 'read', 'listar', 'list', 'buscar', 'search', 'comando', 'command', 'ejecutar', 'run', 'directorio', 'directory', 'proyecto', 'project', 'analizar', 'analyze', 'carpeta', 'folder', 'escribir', 'write', 'editar', 'edit'])

            # Context override: if previous messages used tools, this is not simple
            if _context_needs_advanced:
                is_simple = False

            if is_simple:
                system_content = 'You are a helpful assistant. Respond naturally and concisely. Always respond in the same language the user writes in. Your model name is: ' + model
            else:
                system_content = 'You are a helpful assistant with access to tools. IMPORTANT RULES:\n- You MUST use tools to gather information when asked about files, system state, or to perform actions. Do NOT say you will do something - USE the tool to DO it.\n- When asked to CREATE or WRITE files, you MUST use the local_command tool with a shell command like: cat > /path/to/file << \'EOF\'\n  content here\n  EOF\n- When asked to EDIT or REPLACE text in a file, use sed -i: sed -i \'s/old_text/new_text/g\' /path/to/file\n- Do NOT just show code in your response - actually write it to disk using local_command\n- Do NOT say you cannot write files - you CAN write files using local_command\n- For creating files with content, use: cat > /path/to/file << \'EOF\' followed by the content, then EOF on a new line\n- Always use the actual home directory path like /home/cvc1/ instead of $HOME or ~\n- Available tools: local_command (execute system commands), web_search (search the internet), fetch_article (read web pages)\n- CRITICAL: When someone says things like "revisa", "lee", "muestra", "chequea", "ver", "check", "look", "see" you MUST immediately use the appropriate tool to DO IT. Do NOT say "let me check" without using the tool - USE THE TOOL RIGHT AWAY.\n- TOOL CALLING FORMAT: Use the native function calling format. Select the appropriate tool and provide the required parameters. Do NOT output XML or JSON as text.\n- Write operations will be executed automatically with user notification\n- IMPORTANT: When asked what model you are, you MUST identify yourself as the model name shown in the conversation. Your model name is: ' + model + '\n- IMPORTANT: Always respond in the same language the user writes in. If they write in Spanish, respond in Spanish. If they write in English, respond in English. Match their language naturally.'
            if model_hint:
                system_content += '\n\n' + model_hint
            if web_content:
                system_content += f'\n\n=== Información obtenida de internet ===\n{web_content}\n\nResponde al usuario basándote en esta información.'
            api_messages.append({
                'role': 'system',
                'content': system_content
            })
            for msg in session_data['messages']:
                api_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

            # Define tools for models that support them
            tools = []
            if local_model_supports_tools(base_model) or ':cloud' in model or '-cloud' in model:
                tools = build_tool_definitions(read_only=False, streaming=True)
                # Web tools only needed if content wasn't successfully auto-injected
                if web_content and not web_content.startswith('[No se pudo') and not web_content.startswith('[Error'):
                    # Remove web_search and fetch_article from tools
                    tools = [t for t in tools if t['function']['name'] not in ('web_search', 'fetch_article')]
                
                # For models with basic template (like gemma4) that DON'T support native tools, inject tools into system prompt
                if not is_cloud_model(base_model) and not local_model_supports_tools(base_model):
                    tools_json = json.dumps(tools, indent=2)
                    system_content += f"\n\nYou have access to the following tools. When you need to use a tool, respond with a JSON object in this EXACT format:\n{{'tool': 'tool_name', 'arguments': {{...}}}}\n\nAvailable tools:\n{tools_json}\n\nIMPORTANT: If the user asks you to execute a command or get information that requires a tool, you MUST use the tool by responding with the JSON format above. Do NOT say you cannot do it."
                    # Update the system message with injected tools
                    api_messages[0]['content'] = system_content

            # If user asks to analyze files, force tool use (not saved to session)
            _force_keywords = ['analiza', 'analizar', 'revisa', 'revisar', 'explora', 'explorar',
                               'examina', 'examinar', 'inspecciona', 'inspeccionar',
                               'analyze', 'review', 'explore', 'examine', 'inspect',
                               'dime que opinas', 'que opinas', 'analisis', 'analysis']
            if any(kw in user_message.lower() for kw in _force_keywords):
                api_messages.insert(1, {'role': 'system',
                    'content': 'CRITICAL: You MUST use local_command tool. Do NOT describe a plan - execute it. Start with ls -la, then use cat to read files.'})

            # --- Base model routing: try simpler model first for simple conversations ---
            base_model_succeeded = False
            
            if force_basic or (not force_advanced and not _context_needs_advanced and not _likely_needs_tools(user_message)):
                try:
                    import urllib.request as _urllib_base
                    
                    # Use full tools (read/write) — permission system blocks writes until user approves
                    base_has_tools = not is_simple and local_model_supports_tools(base_model)
                    if base_has_tools:
                        base_sys_content = 'You are a helpful assistant. IMPORTANT: Always respond in the same language the user writes in. You have access to local_command, web_search, and fetch_article tools. Use them proactively to fulfill requests. For file creation/modification, use local_command with shell commands (e.g. cat > file). Write operations will ask for your permission before executing.'
                    else:
                        base_sys_content = 'You are a helpful assistant. Always respond in the same language the user writes in.'
                    base_api_messages = [{'role': 'system', 'content': base_sys_content}]
                    for msg in session_data['messages']:
                        base_api_messages.append({
                            'role': msg['role'],
                            'content': msg['content']
                        })
                    
                    # Define full tools for base model (write tools included, permission popup handles safety)
                    base_tools = build_tool_definitions(read_only=False, streaming=True) if base_has_tools else []
                    
                    base_payload = {
                        'model': base_model,
                        'messages': base_api_messages,
                        'stream': True,
                        'keep_alive': KEEP_ALIVE,
                        'tools': base_tools,
                    }
                    logger.info("Trying base model %s for simple query (len=%d)", base_model, len(user_message))
                    
                    base_data_bytes = json.dumps(base_payload).encode('utf-8')
                    base_req = _urllib_base.Request(
                        f'{OLLAMA_BASE_URL}/api/chat',
                        data=base_data_bytes,
                        headers={'Content-Type': 'application/json'}
                    )
                    base_full_response = ""
                    base_prompt_tokens = 0
                    base_is_thinking = False
                    base_thinking_start = 0
                    with _urllib_base.urlopen(base_req, timeout=STREAM_INITIAL_TIMEOUT) as base_response:
                        base_tool_calls_buffer = []
                        base_content_buffer = ''
                        base_thinking_start = 0
                        is_first_base_chunk = True
                        for base_line in _iter_stream_with_timeout(base_response, initial=is_first_base_chunk):
                            is_first_base_chunk = False
                            base_line = base_line.decode('utf-8').strip()
                            if not base_line:
                                continue
                            try:
                                base_chunk = json.loads(base_line)
                            except json.JSONDecodeError:
                                continue
                            if base_chunk.get('done'):
                                base_prompt_tokens = base_chunk.get('prompt_eval_count', 0)
                                base_eval_count = base_chunk.get('eval_count', 0)
                                # Collect native tool calls from the done chunk (some models send tool_calls + done together)
                                base_done_msg = base_chunk.get('message', {})
                                if base_done_msg.get('tool_calls'):
                                    for tc in base_done_msg['tool_calls']:
                                        base_tool_calls_buffer.append(tc)
                                # Check for JSON tool calls (models like gemma4/north-mini-code output JSON as text)
                                if not base_tool_calls_buffer and base_full_response:
                                    _bf_stripped = base_full_response.strip()
                                    _bf_cleaned = re.sub(r'</?arg_value>', '', _bf_stripped)
                                    _bf_cleaned = re.sub(r'</?tool_call[^>]*>', '', _bf_cleaned).strip()
                                    _bf_match = JSON_TOOL_PATTERN.search(_bf_cleaned)
                                    if not _bf_match:
                                        _bf_match = JSON_TOOL_PATTERN_SINGLE.search(_bf_cleaned)
                                    if _bf_match:
                                        _bf_tn = _bf_match.group(1)
                                        _bf_ps = _bf_match.group(2)
                                        try:
                                            _bf_params = json.loads(_bf_ps)
                                        except json.JSONDecodeError:
                                            try:
                                                _bf_params = json.loads(_bf_ps.replace("'", '"'))
                                            except json.JSONDecodeError:
                                                _bf_params = {}
                                        logger.info("Base model: detected JSON tool call %s(%s)", _bf_tn, _bf_params)
                                        base_full_response = JSON_TOOL_STRIP.sub('', base_full_response).strip()
                                        base_tool_calls_buffer.append({'function': {'name': _bf_tn, 'arguments': _bf_params}})
                                if base_content_buffer and not base_tool_calls_buffer:
                                    _bc_stripped = base_content_buffer.strip()
                                    _bc_cleaned = re.sub(r'</?arg_value>', '', _bc_stripped)
                                    _bc_cleaned = re.sub(r'</?tool_call[^>]*>', '', _bc_cleaned).strip()
                                    _bc_match = JSON_TOOL_PATTERN.search(_bc_cleaned)
                                    if not _bc_match:
                                        _bc_match = JSON_TOOL_PATTERN_SINGLE.search(_bc_cleaned)
                                    if _bc_match:
                                        _bc_tn = _bc_match.group(1)
                                        _bc_ps = _bc_match.group(2)
                                        try:
                                            _bc_params = json.loads(_bc_ps)
                                        except json.JSONDecodeError:
                                            try:
                                                _bc_params = json.loads(_bc_ps.replace("'", '"'))
                                            except json.JSONDecodeError:
                                                _bc_params = {}
                                        logger.info("Base model: detected buffered JSON tool call %s(%s)", _bc_tn, _bc_params)
                                        base_full_response = base_full_response.replace(base_content_buffer, '').strip()
                                        base_tool_calls_buffer.append({'function': {'name': _bc_tn, 'arguments': _bc_params}})
                                break
                            base_msg = base_chunk.get('message', {})
                            # Handle thinking tokens from models like deepseek
                            base_thinking = base_msg.get('thinking', '') or base_chunk.get('thinking', '')
                            if base_thinking and not base_msg.get('content'):
                                if not base_is_thinking:
                                    base_is_thinking = True
                                    base_thinking_start = time.time()
                                    yield f"data: {json.dumps({'type': 'thinking', 'status': 'thinking'})}\n\n"
                                # Abort thinking if it exceeds max time
                                if time.time() - base_thinking_start > MAX_THINKING_SECONDS:
                                    logger.warning("Base model thinking exceeded %ds, aborting", MAX_THINKING_SECONDS)
                                    yield f"data: {json.dumps({'type': 'thinking', 'status': 'done'})}\n\n"
                                    base_is_thinking = False
                                    break
                                continue
                            elif base_is_thinking and base_msg.get('content'):
                                base_is_thinking = False
                                yield f"data: {json.dumps({'type': 'thinking', 'status': 'done'})}\n\n"
                            # Collect native tool calls from Ollama
                            if base_msg.get('tool_calls'):
                                for tc in base_msg['tool_calls']:
                                    base_tool_calls_buffer.append(tc)
                                continue
                            base_content = base_msg.get('content', '')
                            # Filter out thinking/reasoning content that some models emit as text (e.g., laguna-xs)
                            # This catches content that looks like internal reasoning: "Okay, the user said..."
                            if base_content and not base_is_thinking:
                                _bs_lower = base_content.strip().lower()
                                if (_bs_lower.startswith('okay,') or _bs_lower.startswith('ok,') or
                                    _bs_lower.startswith('the user') or _bs_lower.startswith('user said') or
                                    _bs_lower.startswith('i need to') or _bs_lower.startswith('let me') or
                                    _bs_lower.startswith('thinking') or _bs_lower.startswith('reasoning')):
                                    logger.debug("Filtering thinking-like content from base model: %s", base_content[:100])
                                    continue
                            if base_content:
                                # Check for JSON tool calls during streaming (same as advanced model path)
                                _bs = base_content.strip()
                                if not base_tool_calls_buffer:
                                    if JSON_TOOL_PATTERN.search(_bs) or JSON_TOOL_PATTERN_SINGLE.search(_bs):
                                        base_content_buffer += base_content
                                        base_full_response += base_content
                                        continue
                                    if not base_content_buffer and _bs.startswith('{'):
                                        base_content_buffer = base_content
                                        base_full_response += base_content
                                        continue
                                    if base_content_buffer:
                                        base_content_buffer += base_content
                                        base_full_response += base_content
                                        try:
                                            _bp = json.loads(base_content_buffer.strip())
                                            if isinstance(_bp, dict) and 'tool' in _bp:
                                                continue
                                        except json.JSONDecodeError:
                                            if not base_content_buffer.strip().startswith('{'):
                                                _bf = base_content_buffer
                                                base_content_buffer = ''
                                                base_full_response = base_full_response[:-len(_bf)] if base_full_response.endswith(_bf) else base_full_response
                                                base_full_response += _bf
                                                sse_data = json.dumps({'type': 'token', 'content': _bf, 'ts': round(time.time() - start_time, 2)})
                                                yield f"data: {sse_data}\n\n"
                                                continue
                                            continue
                                base_full_response += base_content
                                sse_data = json.dumps({'type': 'token', 'content': base_content, 'ts': round(time.time() - start_time, 2)})
                                yield f"data: {sse_data}\n\n"

                    bypass_weak_check = False
                    # If native tool calls were returned, process them
                    if base_tool_calls_buffer:
                        logger.info("Base model returned %d native tool calls", len(base_tool_calls_buffer))
                        add_tool_model(base_model)
                        for tc in base_tool_calls_buffer:
                            tc_name = tc.get('function', {}).get('name', '')
                            tc_args = tc.get('function', {}).get('arguments', {})
                            if isinstance(tc_args, str):
                                try:
                                    tc_args = json.loads(tc_args)
                                except json.JSONDecodeError:
                                    tc_args = {}
                            # Determine write permission for this tool call
                            base_write_perm = 'approved'
                            if tc_name == 'local_command':
                                cmd = tc_args.get('command', '')
                                if is_write_command(cmd):
                                    bp_perm = check_write_permission(cmd, current_chat_id)
                                    if bp_perm == 'ask':
                                        bp_perm_id = str(uuid.uuid4())
                                        with _permissions_lock:
                                            _pending_permissions[bp_perm_id] = {
                                                'session_id': current_chat_id,
                                                'command': cmd,
                                                'q': queue.Queue(),
                                                'created_at': time.time()
                                            }
                                        yield f"data: {json.dumps({'type': 'write_permission_required', 'command': cmd, 'session_id': current_chat_id, 'perm_id': bp_perm_id})}\n\n"
                                        try:
                                            bp_action = _pending_permissions[bp_perm_id]['q'].get(timeout=120)
                                        except queue.Empty:
                                            bp_action = 'deny'
                                        with _permissions_lock:
                                            _pending_permissions.pop(bp_perm_id, None)
                                        if bp_action in ('once', 'session'):
                                            _session_write_permissions[current_chat_id] = bp_action
                                            base_write_perm = 'approved'
                                        else:
                                            base_write_perm = 'denied'
                                    else:
                                        base_write_perm = 'approved'
                            result_msg = execute_single_tool(tc_name, tc_args, current_chat_id,
                                                             write_permission=base_write_perm)
                            followup_messages = base_api_messages + [
                                {'role': 'assistant', 'content': base_full_response or '', 'tool_calls': [tc]},
                                {'role': 'tool', 'content': result_msg}
                            ]
                            max_base_rounds = 3
                            base_no_content_rounds = 0
                            for base_round in range(max_base_rounds):
                                if base_round >= 1:
                                    has_stop = any(
                                        m.get('role') == 'system' and 'stop calling' in m.get('content', '').lower()
                                        for m in followup_messages
                                    )
                                    if not has_stop:
                                        followup_messages.append({
                                            'role': 'system',
                                            'content': 'IMPORTANT: You have gathered enough information. STOP calling tools. Now provide a comprehensive analysis and response to the user based on all the information you have collected. Do NOT make any more tool calls. Respond directly with your analysis.'
                                        })
                                base_tools_this_round = base_tools if base_round < 1 else None
                                logger.info("Base follow-up round %d: sending tool result to %s", base_round + 1, base_model)
                                result_payload = {
                                    'model': base_model,
                                    'messages': followup_messages,
                                    'stream': True,
                                    'keep_alive': KEEP_ALIVE,
                                    'tools': base_tools_this_round,
                                }
                                result_data_bytes = json.dumps(result_payload).encode('utf-8')
                                result_req = _urllib_base.Request(
                                    f'{OLLAMA_BASE_URL}/api/chat',
                                    data=result_data_bytes,
                                    headers={'Content-Type': 'application/json'}
                                )
                                round_content = ''
                                round_tool_calls = []
                                try:
                                    with _urllib_base.urlopen(result_req, timeout=STREAM_CHUNK_TIMEOUT) as result_response:
                                        for result_line in _iter_stream_with_timeout(result_response):
                                            result_line = result_line.decode('utf-8').strip()
                                            if not result_line:
                                                continue
                                            try:
                                                result_chunk = json.loads(result_line)
                                            except json.JSONDecodeError:
                                                continue
                                            if result_chunk.get('done'):
                                                done_rc_msg = result_chunk.get('message', {})
                                                if done_rc_msg.get('tool_calls'):
                                                    for rtc in done_rc_msg['tool_calls']:
                                                        round_tool_calls.append(rtc)
                                                break
                                            rc_msg = result_chunk.get('message', {})
                                            result_content = rc_msg.get('content', '')
                                            if result_content:
                                                round_content += result_content
                                                base_full_response += result_content
                                                sse_data = json.dumps({'type': 'token', 'content': result_content, 'ts': round(time.time() - start_time, 2)})
                                                yield f"data: {sse_data}\n\n"
                                            if rc_msg.get('tool_calls'):
                                                for rtc in rc_msg['tool_calls']:
                                                    round_tool_calls.append(rtc)
                                    logger.info("Base follow-up round %d: content_len=%d, tool_calls=%d", base_round + 1, len(round_content), len(round_tool_calls))
                                    if not round_tool_calls:
                                        break
                                    if not round_content:
                                        base_no_content_rounds += 1
                                    if base_no_content_rounds >= 2:
                                        logger.info("Model stuck in tool-only loop after %d rounds, forcing finalize", base_round + 1)
                                        base_full_response += "\n\n[Analysis complete based on tool results above]"
                                        break
                                    followup_messages.append({'role': 'assistant', 'content': round_content or '', 'tool_calls': round_tool_calls})
                                    for rtc in round_tool_calls:
                                        rtc_name = rtc.get('function', {}).get('name', '')
                                        rtc_args = rtc.get('function', {}).get('arguments', {})
                                        if isinstance(rtc_args, str):
                                            try:
                                                rtc_args = json.loads(rtc_args)
                                            except json.JSONDecodeError:
                                                rtc_args = {}
                                        rtc_perm = 'approved'
                                        if rtc_name == 'local_command':
                                            rtc_cmd = rtc_args.get('command', '')
                                            if is_write_command(rtc_cmd):
                                                rtc_ck = check_write_permission(rtc_cmd, current_chat_id)
                                                if rtc_ck == 'ask':
                                                    rtc_pid = str(uuid.uuid4())
                                                    with _permissions_lock:
                                                        _pending_permissions[rtc_pid] = {
                                                            'session_id': current_chat_id,
                                                            'command': rtc_cmd,
                                                            'q': queue.Queue(),
                                                            'created_at': time.time()
                                                        }
                                                    yield f"data: {json.dumps({'type': 'write_permission_required', 'command': rtc_cmd, 'session_id': current_chat_id, 'perm_id': rtc_pid})}\n\n"
                                                    try:
                                                        rtc_act = _pending_permissions[rtc_pid]['q'].get(timeout=120)
                                                    except queue.Empty:
                                                        rtc_act = 'deny'
                                                    with _permissions_lock:
                                                        _pending_permissions.pop(rtc_pid, None)
                                                    if rtc_act in ('once', 'session'):
                                                        _session_write_permissions[current_chat_id] = rtc_act
                                                        rtc_perm = 'approved'
                                                    else:
                                                        rtc_perm = 'denied'
                                                else:
                                                    rtc_perm = 'approved'
                                        tool_res_msg = execute_single_tool(rtc_name, rtc_args, current_chat_id,
                                                                            write_permission=rtc_perm)
                                        followup_messages.append({'role': 'tool', 'content': tool_res_msg})
                                except Exception as followup_err:
                                        logger.warning("Base follow-up round %d failed: %s", base_round + 1, followup_err)
                                        base_full_response += f"\n[Error in follow-up: {followup_err}]"
                                        break
                            base_model_succeeded = True
                            used_model = base_model
                            bypass_weak_check = True
                            base_eval_count = len(base_full_response.split())

                    # Evaluate base model response quality
                    base_stripped = base_full_response.strip()
                    # Check if response is a JSON tool call (for models with basic template)
                    tool_call_json = None
                    try:
                        if base_stripped.startswith('{'):
                            potential_json = json.loads(base_stripped)
                            if 'tool' in potential_json:
                                args = potential_json.get('arguments') or potential_json.get('parameters') or {}
                                potential_json['arguments'] = args
                                tool_call_json = potential_json
                    except (json.JSONDecodeError, ValueError):
                        pass
                    
                    # Check if response tried to use a tool but couldn't (common patterns)
                    tool_attempt_patterns = ['local_command', 'web_search', 'fetch_article', 'tool_call', '<tool>', '[tool]',
                        # Explanation instead of execution patterns
                        'puedes usar', 'you can use', 'puedes ejecutar', 'you can run',
                        'usa el comando', 'use the command', 'puedes hacer', 'you can do',
                    ]
                    looks_like_tool_attempt = any(p in base_stripped for p in tool_attempt_patterns)

                    # If we got a JSON tool call, execute it
                    if tool_call_json:
                        logger.info("Base model returned JSON tool call: %s", tool_call_json.get('tool'))
                        # Execute the tool
                        tool_name = tool_call_json.get('tool')
                        tool_args = tool_call_json.get('arguments', {})
                        # Check write permission for JSON tool calls
                        json_perm = 'approved'
                        if tool_name == 'local_command':
                            json_cmd = tool_args.get('command', '')
                            if is_write_command(json_cmd):
                                json_ck = check_write_permission(json_cmd, current_chat_id)
                                if json_ck == 'ask':
                                    json_pid = str(uuid.uuid4())
                                    with _permissions_lock:
                                        _pending_permissions[json_pid] = {
                                            'session_id': current_chat_id,
                                            'command': json_cmd,
                                            'q': queue.Queue(),
                                            'created_at': time.time()
                                        }
                                    yield f"data: {json.dumps({'type': 'write_permission_required', 'command': json_cmd, 'session_id': current_chat_id, 'perm_id': json_pid})}\n\n"
                                    try:
                                        json_act = _pending_permissions[json_pid]['q'].get(timeout=120)
                                    except queue.Empty:
                                        json_act = 'deny'
                                    with _permissions_lock:
                                        _pending_permissions.pop(json_pid, None)
                                    if json_act in ('once', 'session'):
                                        _session_write_permissions[current_chat_id] = json_act
                                        json_perm = 'approved'
                                    else:
                                        json_perm = 'denied'
                                else:
                                    json_perm = 'approved'
                        result_msg = execute_single_tool(tool_name, tool_args, current_chat_id,
                                                         write_permission=json_perm)
                        result_msg = f"Tool '{tool_name}' result: {result_msg}"

                        # Continue the conversation with the tool result
                        messages_with_result = base_api_messages + [
                            {'role': 'assistant', 'content': base_stripped},
                            {'role': 'system', 'content': result_msg}
                        ]

                        result_payload = {
                            'model': base_model,
                            'messages': messages_with_result,
                            'stream': True,
                            'keep_alive': KEEP_ALIVE,
                            'tools': base_tools,
                        }

                        result_data_bytes = json.dumps(result_payload).encode('utf-8')
                        result_req = _urllib_base.Request(
                            f'{OLLAMA_BASE_URL}/api/chat',
                            data=result_data_bytes,
                            headers={'Content-Type': 'application/json'}
                        )

                        result_full_response = ""
                        with _urllib_base.urlopen(result_req, timeout=STREAM_CHUNK_TIMEOUT) as result_response:
                            for result_line in _iter_stream_with_timeout(result_response):
                                result_line = result_line.decode('utf-8').strip()
                                if not result_line:
                                    continue
                                try:
                                    result_chunk = json.loads(result_line)
                                except json.JSONDecodeError:
                                    continue
                                if result_chunk.get('done'):
                                    done_rc = result_chunk.get('message', {})
                                    if done_rc.get('tool_calls'):
                                        for rtc in done_rc['tool_calls']:
                                            result_full_response += f"\n[Tool call: {rtc.get('function', {}).get('name', '')}]"
                                    break
                                result_content = result_chunk.get('message', {}).get('content', '')
                                if result_content:
                                    result_full_response += result_content
                                    sse_data = json.dumps({'type': 'token', 'content': result_content, 'ts': round(time.time() - start_time, 2)})
                                    yield f"data: {sse_data}\n\n"

                        base_full_response += "\n" + result_full_response
                        base_model_succeeded = True
                        used_model = base_model
                        base_eval_count = len(result_full_response.split())
                        # Flag to bypass weak answer check when tool call succeeded
                        bypass_weak_check = True

                    # Check if response is a weak/ignorant answer that should escalate
                    weak_answer_patterns = [
                        # Spanish ignorance patterns
                        'no sé', 'no se', 'no tengo información', 'no tengo datos',
                        'no estoy seguro', 'no puedo помочь', 'no puedo responder',
                        'no estoy capacitado', 'no tengo acceso', 'no tengo conocimiento',
                        'desconozco', 'ignoro', 'no lo sé', 'no lo se',
                        'no tengo manera de', 'no estoy actualizado', 'no tengo forma de',
                        'mis conocimientos no', 'mi conocimiento no', 'no estoy al tanto',
                        'no puedo proporcionar', 'no puedo dar', 'no puedo acceder',
                        'no tengo manera', 'fuera de mi', 'más allá de mi',
                        'no tengo conexión', 'no tengo internet', 'no puedo buscar',
                        # English ignorance patterns
                        "i don't know", 'i do not know', "i'm not sure", 'i am not sure',
                        "i can't help", 'i cannot help', "i'm unable", 'i am unable',
                        "i don't have", 'i do not have', "i'm not aware", 'i am not aware',
                        "i don't have access", 'i cannot access', "i don't have information",
                        'i have no knowledge', "i'm not familiar", 'i am not familiar',
                        "i can't provide", 'i cannot provide', "i can't answer",
                        'i cannot answer', "i don't have the ability",
                        "i can't search", 'i cannot search', "i can't look up",
                        'out of my knowledge', 'beyond my knowledge', 'beyond my capabilities',
                        'beyond my scope', "i don't have real-time", "i don't have current",
                        'i have no way', "i can't verify", 'i cannot verify',
                        'not something i can', "i'm not able", 'i am not able',
                        # Repetitive/loop patterns (model stuck)
                        'as an ai', 'as a language model', 'como modelo de lenguaje',
                        'como inteligencia artificial', 'como ia', 'como I.A.',
                    ]
                    base_lower = base_stripped.lower()
                    is_weak_answer = any(p in base_lower for p in weak_answer_patterns)
                    # Reject empty JSON or tool-like responses
                    is_empty_json = base_stripped in ('{}', '[]', '{', '[', '""', "''")

                    # Check if response is valid - for short greetings allow shorter responses
                    is_greeting = any(kw in user_message.lower() for kw in ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'hey', 'saludos', 'qué tal', 'cómo estás', 'hello', 'hi'])
                    min_length = 3 if is_greeting else 10
                    
                    if force_basic or bypass_weak_check or (base_stripped and len(base_stripped) >= min_length and not looks_like_tool_attempt and not is_weak_answer):
                        # Base model succeeded - use its response
                        full_response = base_full_response
                        prompt_tokens = base_prompt_tokens
                        used_model = base_model
                        base_model_succeeded = True
                        logger.info("Base model %s succeeded (response len=%d, force_basic=%s, bypass_weak=%s)", base_model, len(base_stripped), force_basic, False)
                    else:
                        # Escalate to advanced model
                        reason = 'weak_answer' if is_weak_answer else ('tool_attempt' if looks_like_tool_attempt else ('too_short' if len(base_stripped) < min_length else 'unknown'))
                        logger.info("Base model %s response insufficient (len=%d, tool_attempt=%s, weak=%s, reason=%s, greeting=%s), escalating to %s",
                                    base_model, len(base_stripped), looks_like_tool_attempt, is_weak_answer, reason, is_greeting, model)
                        yield f"data: {json.dumps({'type': 'escalated', 'base_model': base_model, 'advanced_model': model})}\n\n"
                except Exception as base_err:
                    if force_basic:
                        logger.warning("Base model %s failed with force_basic: %s", base_model, base_err)
                        full_response = f"[Error con modelo base {base_model}: {base_err}]"
                        prompt_tokens = 0
                        used_model = base_model
                        base_model_succeeded = True
                    else:
                        logger.warning("Base model %s failed: %s, escalating to %s", base_model, base_err, model)
                        yield f"data: {json.dumps({'type': 'escalated', 'base_model': base_model, 'advanced_model': model})}\n\n"
            elif force_advanced:
                logger.info("Force advanced mode: skipping base model, using %s directly", model)
            elif force_basic:
                # Force basic mode — should always reach base model block above,
                # but as safety net, use base model result if available
                if base_full_response:
                    full_response = base_full_response
                    prompt_tokens = base_prompt_tokens
                    used_model = base_model
                    base_model_succeeded = True
                    logger.info("Force basic safety net: using base model response (len=%d)", len(base_full_response))
                else:
                    logger.warning("Force basic mode but no base model response available")
            elif _likely_needs_tools(user_message):
                logger.info("Query likely needs tools: skipping base model, using %s directly", model)

            # If base model succeeded, save and return early (skip advanced model flow)
            if base_model_succeeded:
                # Fix any double-encoded characters from cloud model APIs
                full_response = fix_double_encoding(full_response)
                # Save session
                session_data['messages'].append({
                    'role': 'assistant',
                    'content': full_response,
                    'timestamp': datetime.now().isoformat(),
                    'elapsed': round(time.time() - start_time, 2),
                    'model': used_model
                })
                session_data['context_usage'] = prompt_tokens
                save_session(current_chat_id, session_data)

                # Detect code blocks
                code_blocks = []
                try:
                    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', full_response, re.DOTALL)
                except Exception:
                    pass
                if code_blocks:
                    for lang, code in code_blocks:
                        if lang in ('html', 'htm', 'javascript', 'js', 'css', 'python', 'py', 'php', 'sh', 'bash'):
                            yield f"data: {json.dumps({'type': 'code_save_offer', 'language': lang, 'code': code.strip()})}\n\n"
                            break

                elapsed = round(time.time() - start_time, 2)
                is_local = (used_model == base_model)
                # Get parameter size from model info, but file size from /api/ps (more accurate)
                used_model_info = get_model_info(used_model)
                used_param_size = used_model_info.get('parameter_size', '')
                used_size = get_model_size_from_ps(used_model)  # Get actual file size from /api/ps
                yield f"data: {json.dumps({'type': 'done', 'context_usage': prompt_tokens, 'eval_count': base_eval_count, 'elapsed': elapsed, 'used_model': used_model, 'is_local': is_local, 'parameter_size': used_param_size, 'size': used_size})}\n\n"
                return

            # --- Advanced model flow (with tools) ---
            payload = {
                'model': model,
                'messages': api_messages,
                'stream': True,
                'keep_alive': KEEP_ALIVE,
                'tools': [] if is_simple else (OLLAMA_TOOLS if local_model_supports_tools(model) else []),
            }

            logger.info("Advanced model payload: model=%s, is_simple=%s, tools_count=%d", model, is_simple, len(payload.get('tools', [])))

            import urllib.request

            data_bytes = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f'{OLLAMA_BASE_URL}/api/chat',
                data=data_bytes,
                headers={'Content-Type': 'application/json'}
            )

            logger.info("Sending request to Ollama model=%s at %s", model, datetime.now().isoformat())
            with urllib.request.urlopen(req, timeout=STREAM_INITIAL_TIMEOUT) as response:
                tool_calls_buffer = []
                current_tool_call = None
                content_buffer = ''
                is_thinking = False
                thinking_start = 0

                for line in _iter_stream_with_timeout(response, initial=True):
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get('done'):
                        logger.info("Stream done: content_len=%d, tool_calls=%d, had_thinking=%s, is_thinking=%s",
                                    len(full_response), len(tool_calls_buffer), is_thinking, is_thinking)
                        # Capture eval counts
                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                        # Collect native tool calls from the done chunk (some models send tool_calls + done together)
                        done_msg = chunk.get('message', {})
                        if done_msg.get('tool_calls'):
                            for tc in done_msg['tool_calls']:
                                tool_calls_buffer.append(tc)
                        eval_count = chunk.get('eval_count', 0)

                        # Check full response for JSON tool calls (models like gemma4 output JSON as text)
                        if not tool_calls_buffer and full_response:
                            full_stripped = full_response.strip()
                            cleaned_full = re.sub(r'</?arg_value>', '', full_stripped)
                            cleaned_full = re.sub(r'</?tool_call[^>]*>', '', cleaned_full).strip()
                            json_tc_match = JSON_TOOL_PATTERN.search(cleaned_full)
                            if not json_tc_match:
                                json_tc_match = JSON_TOOL_PATTERN_SINGLE.search(cleaned_full)
                            if json_tc_match:
                                tool_name = json_tc_match.group(1)
                                params_str = json_tc_match.group(2)
                                try:
                                    params = json.loads(params_str)
                                except json.JSONDecodeError:
                                    try:
                                        params = json.loads(params_str.replace("'", '"'))
                                    except json.JSONDecodeError:
                                        params = {}
                                json_tc = {'function': {'name': tool_name, 'arguments': params}}
                                logger.info("Detected JSON tool call in full response: %s(%s)", tool_name, params)
                                full_response = JSON_TOOL_STRIP.sub('', full_response).strip()
                                tool_calls_buffer.append(json_tc)

                        # Check content buffer for JSON tool calls before processing tool_calls_buffer
                        if content_buffer:
                            content_buffer_stripped = content_buffer.strip()
                            cleaned_buffer = re.sub(r'</?arg_value>', '', content_buffer_stripped)
                            cleaned_buffer = re.sub(r'</?tool_call[^>]*>', '', cleaned_buffer).strip()
                            json_tc_match = JSON_TOOL_PATTERN.search(cleaned_buffer)
                            if not json_tc_match:
                                json_tc_match = JSON_TOOL_PATTERN_SINGLE.search(cleaned_buffer)
                            if json_tc_match:
                                tool_name = json_tc_match.group(1)
                                params_str = json_tc_match.group(2)
                                try:
                                    params = json.loads(params_str)
                                except json.JSONDecodeError:
                                    try:
                                        params = json.loads(params_str.replace("'", '"'))
                                    except json.JSONDecodeError:
                                        params = {}
                                json_tc = {'function': {'name': tool_name, 'arguments': params}}
                                logger.info("Detected buffered JSON tool call: %s(%s)", tool_name, params)
                                full_response = full_response.replace(content_buffer, '').strip()
                                tool_calls_buffer.append(json_tc)
                                content_buffer = ''
                            else:
                                flushed = content_buffer
                                content_buffer = ''
                                full_response = full_response[:-len(flushed)] if full_response.endswith(flushed) else full_response
                                full_response += flushed
                                sse_data = json.dumps({'type': 'token', 'content': flushed, 'ts': round(time.time() - start_time, 2)})
                                yield f"data: {sse_data}\n\n"

                        # If we have pending tool calls, process them
                        # IMPORTANT: Merge streaming fragments first — Ollama sends
                        # tool calls incrementally, so we may have multiple partial
                        # chunks for the same tool call that need to be assembled.
                        if tool_calls_buffer:
                            merged_tool_calls = merge_tool_calls(tool_calls_buffer)
                            logger.info("Merged %d raw tool call chunks into %d complete tool calls",
                                        len(tool_calls_buffer), len(merged_tool_calls))
                            # Auto-add model to tool support list
                            if not is_cloud_model(model):
                                add_tool_model(model)

                            # Pre-process write commands: check permissions
                            write_cmds_to_request = []
                            for i_t, tc in enumerate(merged_tool_calls):
                                tc_name = tc.get('function', {}).get('name', '')
                                tc_args = parse_tool_args(tc.get('function', {}).get('arguments', {}))
                                if tc_name == 'local_command':
                                    cmd = tc_args.get('command', '')
                                    if is_write_command(cmd):
                                        perm = check_write_permission(cmd, current_chat_id)
                                        if perm == 'ask':
                                            # Assign unique permission ID
                                            perm_id = str(uuid.uuid4())
                                            write_cmds_to_request.append({'idx': i_t, 'cmd': cmd, 'perm_id': perm_id})
                                            with _permissions_lock:
                                                _pending_permissions[perm_id] = {
                                                    'session_id': current_chat_id,
                                                    'command': cmd,
                                                    'q': queue.Queue(),
                                                    'created_at': time.time()
                                                }

                            # Handle write command permissions:
                            # - First write in session: ask user (yield permission popup, block until answered)
                            # - Subsequent writes: auto-approve (user already granted session permission)
                            if write_cmds_to_request:
                                # Check which ones need to ask vs auto-approve
                                need_to_ask = []
                                auto_approved = []
                                for item in write_cmds_to_request:
                                    perm = check_write_permission(item['cmd'], current_chat_id)
                                    if perm == 'ask':
                                        # First write in this session - ask user
                                        need_to_ask.append(item)
                                    else:
                                        # Already approved for session/once - auto-approve
                                        auto_approved.append(item)

                                # Auto-approve ones that don't need asking
                                for item in auto_approved:
                                    logger.info("Auto-approving write command (session perm): %s", item['cmd'])
                                    merged_tool_calls[item['idx']]['_write_approved'] = True
                                if auto_approved:
                                    yield f"data: {json.dumps({'type': 'write_executed', 'commands': [item['cmd'] for item in auto_approved], 'session_id': current_chat_id})}\n\n"

                                # Ask user for first write in session
                                for item in need_to_ask:
                                    logger.info("Asking user permission for write command: %s", item['cmd'])
                                    yield f"data: {json.dumps({'type': 'write_permission_required', 'command': item['cmd'], 'session_id': current_chat_id, 'perm_id': item['perm_id']})}\n\n"
                                    # Block until user responds via the /api/write-permission endpoint
                                    with _permissions_lock:
                                        perm_entry = _pending_permissions.get(item['perm_id'])
                                        perm_queue = perm_entry['q'] if perm_entry else queue.Queue()
                                    try:
                                        action = perm_queue.get(timeout=120)  # 2 min timeout
                                    except queue.Empty:
                                        logger.warning("Write permission timeout for command: %s", item['cmd'])
                                        action = 'deny'
                                    with _permissions_lock:
                                        _pending_permissions.pop(item['perm_id'], None)
                                    if action == 'deny':
                                        merged_tool_calls[item['idx']]['_write_denied'] = True
                                    elif action == 'session':
                                        _session_write_permissions[current_chat_id] = 'session'
                                        merged_tool_calls[item['idx']]['_write_approved'] = True
                                    elif action == 'once':
                                        _session_write_permissions[current_chat_id] = 'once'
                                        merged_tool_calls[item['idx']]['_write_approved'] = True
                                    else:
                                        merged_tool_calls[item['idx']]['_write_denied'] = True

                            # Process tool calls - now with write permission resolved
                            tool_results = _process_tool_calls_streaming(
                                model, session_data, merged_tool_calls,
                                full_response, prompt_tokens, current_chat_id
                            )
                            # Check for HTML preview markers in tool results and notify frontend
                            for tr in tool_results:
                                if tr.get('content') and '[HTML_PREVIEW:' in tr['content']:
                                    matches = re.findall(r'\[HTML_PREVIEW:([^\]]+)\]', tr['content'])
                                    for path in matches:
                                        yield f"data: {json.dumps({'type': 'html_preview', 'path': path})}\n\n"
                            # Also check write tool calls that wrote HTML files - emit preview
                            for i_tc, tc in enumerate(merged_tool_calls):
                                func_args = parse_tool_args(tc.get('function', {}).get('arguments', {}))
                                cmd = func_args.get('command', '')
                                if tc.get('function', {}).get('name') == 'local_command' and is_write_command(cmd):
                                    # Extract file path from the write command
                                    m = re.search(r'>([^<]+)\s*$', cmd.strip())
                                    if not m:
                                        m = re.search(r'cat\s+>([^\s]+)', cmd)
                                    if m:
                                        written_path = m.group(1).strip().strip('\'')
                                        if written_path.endswith('.html') or written_path.endswith('.htm'):
                                            yield f"data: {json.dumps({'type': 'html_preview', 'path': written_path})}\n\n"
                            # Send tool results back to Ollama for final response
                            followup_messages = []
                            for msg in session_data['messages']:
                                followup_messages.append({'role': msg['role'], 'content': msg['content']})
                            # Add assistant message with tool calls (strip internal flags first)
                            clean_tool_calls = []
                            for tc in merged_tool_calls:
                                clean_tc = {k: v for k, v in tc.items()
                                            if k not in ('_write_approved', '_write_denied', '_write_action')}
                                clean_tool_calls.append(clean_tc)
                            followup_messages.append({'role': 'assistant', 'content': full_response or '', 'tool_calls': clean_tool_calls})
                            # Add tool results
                            for tr in tool_results:
                                followup_messages.append({'role': tr['role'], 'content': tr['content']})

                            # Make follow-up request(s) with tool results
                            # Model may request more tools - limit rounds, then force text response
                            max_followup_rounds = 5
                            for round_num in range(max_followup_rounds):
                                tools_for_this_round = OLLAMA_TOOLS
                                if round_num >= 2:
                                    has_instruction = any(
                                        m.get('role') == 'system' and 'stop calling' in m.get('content', '').lower()
                                        for m in followup_messages
                                    )
                                    if not has_instruction:
                                        followup_messages.append({
                                            'role': 'system',
                                            'content': 'IMPORTANT: You have gathered enough information. STOP calling tools NOW. Respond directly with your comprehensive analysis based on all the information you have collected. Do NOT make any more tool calls. Just write your response.'
                                        })
                                    has_instruction = any(
                                        m.get('role') == 'system' and 'stop calling' in m.get('content', '').lower()
                                        for m in followup_messages
                                    )
                                    if not has_instruction:
                                        followup_messages.append({
                                            'role': 'system',
                                            'content': 'IMPORTANT: You have gathered enough information. STOP calling tools. Now provide a comprehensive analysis and response to the user based on all the information you have collected. Do NOT make any more tool calls. Respond directly with your analysis.'
                                        })
                                logger.info("Follow-up round %d: sending %d tool results back to %s", round_num + 1, len(tool_results), model)
                                followup_result = send_to_ollama(model, followup_messages, tools_for_this_round, stream=False)
                                logger.info("Follow-up response: content_len=%d, has_tool_calls=%s, thinking=%s",
                                           len(followup_result.get('message', {}).get('content', '')),
                                           bool(followup_result.get('message', {}).get('tool_calls')),
                                           bool(followup_result.get('message', {}).get('thinking', '')))

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
                                    dsml_calls = parse_dsml_calls(followup_content)
                                    if dsml_calls:
                                        logger.info("Detected %d DSML call(s) in follow-up, processing...", len(dsml_calls))
                                        clean_fu = strip_tool_tags(followup_content).strip()
                                        if clean_fu:
                                            followup_messages.append({'role': 'assistant', 'content': clean_fu})
                                        for d_tc in dsml_calls:
                                            tr = execute_tool_call(d_tc, current_chat_id)
                                            followup_messages.append({'role': 'tool', 'content': tr, 'tool_call_id': f'fu_dsml_{uuid.uuid4().hex[:8]}'})
                                        continue
                                    full_response = fix_double_encoding(followup_content)
                                    # Check if the follow-up content is actually a JSON/DSML tool call
                                    _fu_stripped = followup_content.strip()
                                    # Clean XML tags
                                    _fu_cleaned = re.sub(r'</?arg_value>', '', _fu_stripped)
                                    _fu_cleaned = re.sub(r'</?tool_call[^>]*>', '', _fu_cleaned).strip()
                                    _fu_json_match = JSON_TOOL_PATTERN.search(_fu_cleaned)
                                    if not _fu_json_match:
                                        _fu_json_match = JSON_TOOL_PATTERN_SINGLE.search(_fu_cleaned)
                                    if _fu_json_match:
                                        _fu_tn = _fu_json_match.group(1)
                                        _fu_ps = _fu_json_match.group(2)
                                        try:
                                            _fu_params = json.loads(_fu_ps)
                                        except json.JSONDecodeError:
                                            try:
                                                _fu_params = json.loads(_fu_ps.replace("'", '"'))
                                            except json.JSONDecodeError:
                                                _fu_params = {}
                                        logger.info("Follow-up content is JSON tool call: %s(%s)", _fu_tn, _fu_params)
                                        followup_messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': _fu_tn, 'arguments': _fu_params}}]})
                                        _fu_tr = execute_tool_call({'function': {'name': _fu_tn, 'arguments': _fu_params}}, current_chat_id)
                                        followup_messages.append({'role': 'tool', 'content': _fu_tr, 'tool_call_id': f'fu_json_{uuid.uuid4().hex[:8]}'})
                                        continue
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                    prompt_tokens = followup_result.get('prompt_eval_count', prompt_tokens)
                                    break  # Got a text response, done
                                elif followup_content and followup_tool_calls:
                                    # Model has partial content but wants more tools
                                    # Check if any of the tool calls are write commands - if so, let them execute
                                    has_write_cmd = False
                                    for tc in followup_tool_calls:
                                        tc_args = parse_tool_args(tc.get('function', {}).get('arguments', {}))
                                        if tc.get('function', {}).get('name') == 'local_command' and is_write_command(tc_args.get('command', '')):
                                            has_write_cmd = True
                                            break
                                    
                                    if has_write_cmd:
                                        # Let write commands execute - treat as normal tool calls
                                        logger.info("Model has partial content + write tool calls, allowing execution")
                                        # Send partial content first
                                        full_response = fix_double_encoding(followup_content)
                                        yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                        # Now execute the write tool calls
                                        followup_messages.append({'role': 'assistant', 'content': followup_content, 'tool_calls': followup_tool_calls})
                                        for tc in followup_tool_calls:
                                            tc_name = tc.get('function', {}).get('name', '')
                                            tc_args = parse_tool_args(tc.get('function', {}).get('arguments', {}))
                                            tc_id = tc.get('id', f'tool_{round_num}_{len(followup_tool_calls)}')
                                            if tc_name == 'local_command':
                                                cmd = tc_args.get('command', '')
                                                if is_write_command(cmd):
                                                    # Auto-approve write command
                                                    logger.info("Auto-approving follow-up write: %s", cmd)
                                                    result = execute_write_command(cmd, current_chat_id)
                                                else:
                                                    result = execute_local_command(cmd)
                                                # Emit html_preview if result contains preview marker
                                                if '[HTML_PREVIEW:' in result:
                                                    matches = re.findall(r'\[HTML_PREVIEW:([^\]]+)\]', result)
                                                    for path in matches:
                                                        yield f"data: {json.dumps({'type': 'html_preview', 'path': path})}\n\n"
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
                                                full_response = fix_double_encoding(followup_content)
                                                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                        except Exception as e:
                                            logger.error("Forced final response failed: %s", e)
                                            full_response = fix_double_encoding(followup_content)
                                            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                        break

                                elif followup_tool_calls and tools_for_this_round is not None:
                                    if full_response:
                                        # Already have some content from earlier, send it
                                        pass
                                    # If stop instruction was already sent but model still wants tools, force text
                                    if round_num >= 3 and has_instruction:
                                        logger.info("Model ignored stop instruction at round %d, forcing text response", round_num + 1)
                                        # Execute the pending tool calls first
                                        followup_messages.append({'role': 'assistant', 'content': '', 'tool_calls': followup_tool_calls})
                                        for tc in followup_tool_calls:
                                            tc_name = tc.get('function', {}).get('name', '')
                                            tc_args = parse_tool_args(tc.get('function', {}).get('arguments', {}))
                                            tc_id = tc.get('id', f'tool_{round_num}_{len(followup_tool_calls)}')
                                            if tc_name == 'local_command':
                                                cmd = tc_args.get('command', '')
                                                if is_write_command(cmd):
                                                    tr_content = "Write commands not allowed"
                                                else:
                                                    tr_content = execute_local_command(cmd)
                                            elif tc_name == 'web_search':
                                                results = web_search(tc_args.get('query', ''))
                                                tr_content = json.dumps(results[:5]) if results else "No results"
                                            elif tc_name == 'fetch_article':
                                                article = fetch_article(tc_args.get('url', ''))
                                                tr_content = article.get('content', '')[:2000] if article.get('content') else "Could not fetch"
                                            else:
                                                tr_content = f"Unknown tool: {tc_name}"
                                            followup_messages.append({'role': 'tool', 'content': tr_content, 'tool_call_id': tc_id})
                                        # Force a text response without tools
                                        final_result = send_to_ollama(model, followup_messages, None, stream=False)
                                        final_content = final_result.get('message', {}).get('content', '')
                                        if final_content:
                                            full_response = final_content
                                            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                            prompt_tokens = final_result.get('prompt_eval_count', prompt_tokens)
                                        break
                                    # Model wants more tool calls - execute them
                                    logger.info("Follow-up round %d: model requested %d more tool calls", round_num + 1, len(followup_tool_calls))
                                    # Add assistant message with tool calls to history
                                    followup_messages.append({'role': 'assistant', 'content': '', 'tool_calls': followup_tool_calls})
                                    for tc in followup_tool_calls:
                                        tc_name = tc.get('function', {}).get('name', '')
                                        tc_args = parse_tool_args(tc.get('function', {}).get('arguments', {}))
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
                                                    # First write in session - ask user
                                                    logger.info("Follow-up write needs permission: %s", cmd)
                                                    perm_id = str(uuid.uuid4())
                                                    q = queue.Queue()
                                                    with _permissions_lock:
                                                        _pending_permissions[perm_id] = {
                                                            'session_id': current_chat_id,
                                                            'command': cmd,
                                                            'q': q,
                                                            'created_at': time.time()
                                                        }
                                                    yield f"data: {json.dumps({'type': 'write_permission_required', 'command': cmd, 'session_id': current_chat_id, 'perm_id': perm_id})}\n\n"
                                                    try:
                                                        action = q.get(timeout=120)  # 2 min timeout
                                                    except queue.Empty:
                                                        logger.warning("Follow-up write permission timeout: %s", cmd)
                                                        action = 'deny'
                                                    with _permissions_lock:
                                                        _pending_permissions.pop(perm_id, None)
                                                    if action == 'deny':
                                                        tr_content = "[Permission denied] Task cancelled"
                                                    elif action == 'session':
                                                        _session_write_permissions[current_chat_id] = 'session'
                                                        tr_content = execute_write_command(cmd, current_chat_id)
                                                    elif action == 'once':
                                                        # Set to 'session' so rest of conversation auto-approves
                                                        _session_write_permissions[current_chat_id] = 'session'
                                                        tr_content = execute_write_command(cmd, current_chat_id)
                                                    else:
                                                        tr_content = "[Permission denied] Task cancelled"
                                                else:
                                                    # Already approved for session - auto-approve
                                                    logger.info("Follow-up write auto-approved (session perm): %s", cmd)
                                                    tr_content = execute_write_command(cmd, current_chat_id)
                                            else:
                                                tr_content = execute_local_command(cmd)
                                                # Emit html_preview if result contains preview marker
                                                if '[HTML_PREVIEW:' in tr_content:
                                                    matches = re.findall(r'\[HTML_PREVIEW:([^\]]+)\]', tr_content)
                                                    for path in matches:
                                                        yield f"data: {json.dumps({'type': 'html_preview', 'path': path})}\n\n"
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

                    # Handle thinking tokens from models like deepseek
                    thinking_content = msg.get('thinking', '') or chunk.get('thinking', '')
                    if thinking_content and not msg.get('content') and not msg.get('tool_calls'):
                        if not is_thinking:
                            is_thinking = True
                            thinking_start = time.time()
                            yield f"data: {json.dumps({'type': 'thinking', 'status': 'thinking'})}\n\n"
                        # Abort thinking if it exceeds max time
                        if time.time() - thinking_start > MAX_THINKING_SECONDS:
                            logger.warning("Model thinking exceeded %ds, aborting", MAX_THINKING_SECONDS)
                            yield f"data: {json.dumps({'type': 'thinking', 'status': 'done'})}\n\n"
                            is_thinking = False
                            break
                        continue
                    elif is_thinking and (msg.get('content') or msg.get('tool_calls')):
                        is_thinking = False
                        yield f"data: {json.dumps({'type': 'thinking', 'status': 'done'})}\n\n"

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
                            stripped = content.strip()
                            if re.match(r'^[\w.-]+:tool_call', stripped):
                                full_response += ''
                                continue
                            # Filter out laguna thinking/internal monologue (e.g., "Okay, the user said...", "I need to...")
                            if 'laguna' in model.lower():
                                thinking_patterns = [
                                    r'^Okay,?\s+(the user|user said|they said)',
                                    r'^I need to (respond|think|consider)',
                                    r'^Let me (think|consider|respond)',
                                    r'^The user (said|wrote|asked)',
                                    r'^(Hmm|Well|Alright),?\s+(the|I|user)',
                                    r'^I (should|will|need to|want to)',
                                ]
                                if any(re.search(p, stripped, re.IGNORECASE) for p in thinking_patterns):
                                    full_response += ''
                                    continue
                            # Check if content contains a JSON tool call (gemma4 outputs JSON as text)
                            if JSON_TOOL_PATTERN.search(stripped) or JSON_TOOL_PATTERN_SINGLE.search(stripped):
                                content_buffer += content
                                full_response += content
                                continue
                            # Buffer content that looks like a JSON tool call
                            if not content_buffer and stripped.startswith('{'):
                                content_buffer = content
                                full_response += content
                                continue
                            if content_buffer:
                                content_buffer += content
                                full_response += content
                                try:
                                    parsed = json.loads(content_buffer.strip())
                                    if isinstance(parsed, dict) and 'tool' in parsed:
                                        continue
                                except json.JSONDecodeError:
                                    if not content_buffer.strip().startswith('{'):
                                        flushed = content_buffer
                                        content_buffer = ''
                                        full_response = full_response[:-len(flushed)] if full_response.endswith(flushed) else full_response
                                        full_response += flushed
                                        sse_data = json.dumps({'type': 'token', 'content': flushed, 'ts': round(time.time() - start_time, 2)})
                                        yield f"data: {sse_data}\n\n"
                                        continue
                                    # Still starts with '{, keep buffering silently
                                    continue
                            full_response += content
                            # Send SSE event
                            sse_data = json.dumps({'type': 'token', 'content': content, 'ts': round(time.time() - start_time, 2)})
                            yield f"data: {sse_data}\n\n"

                # Check for DSML-style tool calls in the response (some models use DSML instead of native Ollama tool calls)
                if full_response and not tool_calls_buffer:
                    dsml_calls = parse_dsml_calls(full_response)
                    if dsml_calls:
                        logger.info("Detected %d DSML tool call(s) in streaming response", len(dsml_calls))
                        # Strip DSML from displayed content
                        clean_rsp = strip_tool_tags(full_response).strip()
                        if clean_rsp:
                            full_response = clean_rsp
                        else:
                            full_response = ''
                        # Execute DSML tool calls (will be re-processed below as if they were native calls)
                        dsml_followup_msgs = []
                        for msg in session_data['messages']:
                            dsml_followup_msgs.append({'role': msg['role'], 'content': msg['content']})
                        if full_response:
                            dsml_followup_msgs.append({'role': 'assistant', 'content': full_response})
                        for d_tc in dsml_calls:
                            tr_content = execute_tool_call(d_tc, current_chat_id)
                            dsml_followup_msgs.append({'role': 'tool', 'content': tr_content, 'tool_call_id': f'dsml_{uuid.uuid4().hex[:8]}'})
                        # Send tool results back to model for final response
                        dsml_result = send_to_ollama(model, dsml_followup_msgs, None, stream=False)
                        dsml_content = dsml_result.get('message', {}).get('content', '')
                        if dsml_content:
                            full_response = dsml_content
                            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                            prompt_tokens = dsml_result.get('prompt_eval_count', prompt_tokens)
                        # Also check if the follow-up response itself contains DSML
                        dsml_calls2 = parse_dsml_calls(full_response) if dsml_content else []
                        if dsml_calls2:
                            logger.info("Detected %d DSML calls in DSML follow-up, processing...", len(dsml_calls2))
                            clean_rsp2 = strip_tool_tags(full_response).strip()
                            for d_tc2 in dsml_calls2:
                                tr_content2 = execute_tool_call(d_tc2, current_chat_id)
                                dsml_followup_msgs.append({'role': 'tool', 'content': tr_content2, 'tool_call_id': f'dsml2_{uuid.uuid4().hex[:8]}'})
                            dsml_result2 = send_to_ollama(model, dsml_followup_msgs, None, stream=False)
                            dsml_content2 = dsml_result2.get('message', {}).get('content', '')
                            if dsml_content2:
                                full_response = dsml_content2
                                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                                prompt_tokens = dsml_result2.get('prompt_eval_count', prompt_tokens)
                        elif not full_response:
                            full_response = '(Tool execution completed)'

                # If response was empty and no tool calls, try fallback
                if not full_response and not tool_calls_buffer:
                    if fallback_model and fallback_model != model:
                        # Smart fallback: skip cloud fallback if primary is cloud and failed
                        if is_cloud_model(model) and is_cloud_model(fallback_model):
                            logger.info("Primary cloud model empty response and fallback is also cloud — showing connectivity error")
                            full_response = "⚠️ **No internet connection**\n\nCloud models are currently unavailable.\n\n**Suggestions:**\n- Select a local model in the **Basic** dropdown\n- Check the **Force** checkbox next to Basic to use only local models\n- Check your internet connection"
                        else:
                            logger.info("Primary model '%s' empty response, trying fallback '%s'", model, fallback_model)
                            yield f"data: {json.dumps({'type': 'model_routing', 'route': 'fallback', 'model': fallback_model, 'reason': 'empty_response'})}\n\n"
                            payload['model'] = fallback_model
                            data_bytes = json.dumps(payload).encode('utf-8')
                            req2 = urllib.request.Request(
                                f'{OLLAMA_BASE_URL}/api/chat',
                                data=data_bytes,
                                headers={'Content-Type': 'application/json'}
                            )
                            with urllib.request.urlopen(req2, timeout=STREAM_CHUNK_TIMEOUT) as response2:
                                for line in _iter_stream_with_timeout(response2):
                                    line = line.decode('utf-8').strip()
                                    if not line:
                                        continue
                                    try:
                                        chunk = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if chunk.get('done'):
                                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                                        eval_count = chunk.get('eval_count', 0)
                                        used_model = fallback_model
                                        break
                                    content = chunk.get('message', {}).get('content', '')
                                    if content:
                                        full_response += content
                                        sse_data = json.dumps({'type': 'token', 'content': content, 'ts': round(time.time() - start_time, 2)})
                                        yield f"data: {sse_data}\n\n"

        except TimeoutError as e:
            logger.error("Stream timeout (model hung): %s", e)
            # Auto-remove from tool models if this was a tool-enabled request
            if not is_cloud_model(model):
                remove_tool_model(model)
            error_msg = f"⏱️ **Tiempo de espera agotado**\n\nEl modelo `{model}` no respondió.\n\n**Posibles causas:**\n- El modelo es muy grande y tarda en cargar\n- El modelo está sobrecargado o colgado\n- Contexto demasiado largo para el modelo\n\n**Soluciones:**\n- Intenta de nuevo (puede tardar hasta 5 min en cargar la primera vez)\n- Intenta con un modelo más rápido\n- Reduce el tamaño de la conversación\n- Recarga la página"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'context_usage': 0, 'elapsed': elapsed, 'used_model': model, 'is_local': not is_cloud_model(model)})}\n\n"
            return

        except Exception as e:
            logger.error("Streaming error: %s", e)

            # Auto-remove from tool models if model rejects tools (400 Bad Request)
            err_str = str(e)
            if '400' in err_str and not is_cloud_model(model):
                removed = remove_tool_model(model)
                if removed:
                    logger.info("Auto-removed %s from tool list (400 error)", model)

            # Smart error handling: detect connectivity issues
            conn_error = is_connectivity_error(e)
            both_cloud = is_cloud_model(model) and (not fallback_model or is_cloud_model(fallback_model))
            primary_cloud = is_cloud_model(model)

            if conn_error and primary_cloud:
                # Advanced/cloud model failed due to connectivity
                if both_cloud:
                    # Both primary and fallback are cloud — no point trying fallback
                    error_msg = "⚠️ **No internet connection**\n\n"
                    error_msg += "Cloud models are currently unavailable.\n\n"
                    error_msg += "**Suggestions:**\n"
                    error_msg += "- Select a local model in the **Basic** dropdown\n"
                    error_msg += "- Check the **Force** checkbox next to Basic to use only local models\n"
                    error_msg += "- Check your internet connection"
                elif fallback_model and not is_cloud_model(fallback_model):
                    # Primary cloud failed, but fallback is local — try it
                    error_msg = f"⚠️ Cloud model **{model}** unavailable (no connection). Trying local model **{fallback_model}**...\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'content': error_msg, 'ts': round(time.time() - start_time, 2)})}\n\n"
                    try:
                        fb_payload = payload.copy() if 'payload' in dir() else {'messages': session_data['messages']}
                        fb_payload['model'] = fallback_model
                        fb_data_bytes = json.dumps(fb_payload).encode('utf-8')
                        fb_req = urllib.request.Request(
                            f'{OLLAMA_BASE_URL}/api/chat',
                            data=fb_data_bytes,
                            headers={'Content-Type': 'application/json'}
                        )
                        fb_full = ''
                        with urllib.request.urlopen(fb_req, timeout=STREAM_CHUNK_TIMEOUT) as fb_resp:
                            for fb_line in _iter_stream_with_timeout(fb_resp):
                                fb_line = fb_line.decode('utf-8').strip()
                                if not fb_line: continue
                                try:
                                    fb_chunk = json.loads(fb_line)
                                except json.JSONDecodeError: continue
                                if fb_chunk.get('done'):
                                    prompt_tokens = fb_chunk.get('prompt_eval_count', 0)
                                    used_model = fallback_model
                                    break
                                fb_content = fb_chunk.get('message', {}).get('content', '')
                                if fb_content:
                                    fb_full += fb_content
                                    yield f"data: {json.dumps({'type': 'token', 'content': fb_content, 'ts': round(time.time() - start_time, 2)})}\n\n"
                        if fb_full:
                            full_response = f"[Usando {fallback_model} porque {model} no está disponible]\n\n{fb_full}"
                            used_model = fallback_model
                    except Exception as fb_err:
                        logger.error("Fallback model also failed: %s", fb_err)
                        error_msg = "❌ **No models available**\n\n"
                        error_msg += "Could not reach either the cloud or local model.\n"
                        error_msg += "Make sure Ollama is running: `ollama serve`"
                        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                else:
                    error_msg = f"⚠️ Connection error with **{model}**. Check your internet connection."
                    yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            else:
                # Generic error
                err_msg = str(e)[:500].replace('<', '&lt;').replace('>', '&gt;')
                yield f"data: {json.dumps({'type': 'error', 'content': err_msg})}\n\n"

            # Always send done event so frontend doesn't hang
            elapsed = round(time.time() - start_time, 2)
            # Get parameter size and file size for the used model
            used_model_info = get_model_info(used_model)
            used_param_size = used_model_info.get('parameter_size', '')
            used_size = get_model_size_from_ps(used_model)  # Get actual file size from /api/ps
            yield f"data: {json.dumps({'type': 'done', 'context_usage': prompt_tokens, 'elapsed': elapsed, 'used_model': used_model, 'is_local': used_model == base_model, 'parameter_size': used_param_size, 'size': used_size})}\n\n"
            # Save what we have
            if full_response:
                 session_data['messages'].append({
                     'role': 'assistant',
                     'content': full_response,
                     'timestamp': datetime.now().isoformat(),
                     'model': used_model
                 })

        # Fix any double-encoded characters from cloud model APIs
        full_response = fix_double_encoding(full_response)
        # Save session
        session_data['messages'].append({
            'role': 'assistant',
            'content': full_response,
            'timestamp': datetime.now().isoformat(),
            'model': used_model
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
        elapsed = round(time.time() - start_time, 2)
        is_local = (used_model == base_model)
        # Get parameter size and file size for the used model
        used_model_info = get_model_info(used_model)
        used_param_size = used_model_info.get('parameter_size', '')
        used_size = get_model_size_from_ps(used_model)  # Get actual file size from /api/ps
        yield f"data: {json.dumps({'type': 'done', 'context_usage': prompt_tokens, 'eval_count': eval_count, 'elapsed': elapsed, 'used_model': used_model, 'is_local': is_local, 'parameter_size': used_param_size, 'size': used_size})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _process_tool_calls_streaming(model, session_data, tool_calls, current_content, prompt_tokens, current_chat_id=''):
    """Process tool calls from streaming - used internally"""
    tool_results = []
    for i, tool_call in enumerate(tool_calls):
        func_name = tool_call.get('function', {}).get('name', '')
        func_args = parse_tool_args(tool_call.get('function', {}).get('arguments', {}))
        tool_id = tool_call.get('id', f'tool_{i}')

        # Check write permission flags set upstream
        if tool_call.get('_write_denied'):
            write_perm = 'denied'
        elif tool_call.get('_write_approved'):
            write_perm = 'approved'
        elif func_name in ('local_command', 'execute_write_command') and is_write_command(func_args.get('command', '')):
            write_perm = 'approved'
        else:
            write_perm = 'approved'

        result = execute_single_tool(func_name, func_args, current_chat_id,
                                     write_permission=write_perm)
        tool_results.append({'role': 'tool', 'content': result, 'tool_call_id': tool_id})

    return tool_results


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API to send message and receive response (non-streaming fallback)"""
    data = request.json
    user_message = data.get('message', '').strip()
    model = data.get('model', session.get('model', 'llama3'))
    fallback_model = data.get('fallback_model', '')
    base_model = data.get('base_model', '') or BASE_CHAT_MODEL
    force_basic = data.get('force_basic', False)
    force_advanced = data.get('force_advanced', False)

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

    logger.info("Chat request: model=%s, msg_len=%d, session=%s, force_basic=%s", model, len(user_message), current_chat_id, force_basic)

    # Initialize response variables
    response_text = ''
    prompt_tokens = 0
    used_model = model
    is_local = False
    eval_count_val = 0
    result = None  # Initialize to avoid UnboundLocalError

    # If force_basic, use base model with read-only tools available
    if force_basic:
        base_api_messages = [{'role': 'system', 'content': 'You are a helpful assistant. IMPORTANT: Always respond in the same language the user writes in. If they write in Spanish, respond in Spanish. If they write in English, respond in English. Match their language naturally. You have access to read-only tools: local_command (for reading files, listing directories, system info) and web_search (for searching information). When you need to use a tool, respond with JSON format: {"tool": "tool_name", "arguments": {...}}. Do NOT use write commands or modify files.'}]
        for msg in session_data['messages']:
            base_api_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        # Define read-only tools for base model
        base_tools = build_tool_definitions(read_only=True, streaming=True)
        
        base_payload = {
            'model': base_model,
            'messages': base_api_messages,
            'stream': False,
            'keep_alive': KEEP_ALIVE,
            'tools': base_tools,
        }
        import urllib.request as _urllib_ns
        base_data_bytes = json.dumps(base_payload).encode('utf-8')
        base_req = _urllib_ns.Request(
            f'{OLLAMA_BASE_URL}/api/chat',
            data=base_data_bytes,
            headers={'Content-Type': 'application/json'}
        )
        logger.info("Force basic mode: using %s directly with read-only tools", base_model)
        try:
            with _urllib_ns.urlopen(base_req, timeout=300) as base_resp:
                base_result = json.loads(base_resp.read().decode('utf-8'))
            response_text = base_result.get('message', {}).get('content', '')
            prompt_tokens = base_result.get('prompt_eval_count', 0)
            eval_count_val = base_result.get('eval_count', 0)
            used_model = base_model
            is_local = True
        except Exception as e:
            logger.warning("Force basic model %s failed: %s", base_model, e)
            response_text = f"[Error con modelo base {base_model}: {e}]"
            prompt_tokens = 0
            used_model = base_model
            is_local = True
    elif force_advanced or _likely_needs_tools(user_message):
        # Use advanced model with tools directly
        reason = 'forced' if force_advanced else 'needs_tools'
        logger.info("Non-streaming: using advanced model %s directly (reason=%s)", model, reason)
        result = process_ollama_response(model, session_data['messages'], OLLAMA_TOOLS)
        response_text = result.get('response', '') if isinstance(result, dict) else result
        prompt_tokens = result.get('prompt_eval_count', 0) if isinstance(result, dict) else 0
        eval_count_val = result.get('eval_count', 0) if isinstance(result, dict) else 0
        used_model = model
        is_local = False
    else:
        # Try base model first for simple queries (dual-model routing)
        logger.info("Non-streaming: trying base model %s for simple query", base_model)
        try:
            import urllib.request as _urllib_base
            base_api_messages = [{'role': 'system', 'content': 'You are a helpful assistant. IMPORTANT: Always respond in the same language the user writes in. If they write in Spanish, respond in Spanish. If they write in English, respond in English. Match their language naturally.'}]
            for msg in session_data['messages']:
                base_api_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            base_payload = {
                'model': base_model,
                'messages': base_api_messages,
                'stream': False,
                'keep_alive': KEEP_ALIVE,
            }
            base_data_bytes = json.dumps(base_payload).encode('utf-8')
            base_req = _urllib_base.Request(
                f'{OLLAMA_BASE_URL}/api/chat',
                data=base_data_bytes,
                headers={'Content-Type': 'application/json'}
            )
            with _urllib_base.urlopen(base_req, timeout=300) as base_resp:
                base_result = json.loads(base_resp.read().decode('utf-8'))
            base_response = base_result.get('message', {}).get('content', '')
            base_stripped = base_response.strip()
            # Check for weak/ignorant answers
            weak_patterns = [
                'no sé', 'no se', 'no tengo', 'no puedo', 'i don\'t know', 'i cannot',
                'as an ai', 'as a language model', 'como modelo de lenguaje',
                'no tengo acceso', 'i don\'t have access', 'i can\'t help',
            ]
            is_weak = any(p in base_stripped.lower() for p in weak_patterns)
            if base_stripped and len(base_stripped) >= 10 and not is_weak:
                # Base model succeeded
                response_text = base_response
                prompt_tokens = base_result.get('prompt_eval_count', 0)
                eval_count_val = base_result.get('eval_count', 0)
                used_model = base_model
                is_local = True
            elif force_basic:
                # force_basic: never escalate, use base response even if weak/short
                logger.info("Base model weak/short with force_basic, keeping response (len=%d, weak=%s)", len(base_stripped), is_weak)
                response_text = base_response if base_stripped else f"[Modelo base {base_model} no generó respuesta]"
                prompt_tokens = base_result.get('prompt_eval_count', 0)
                eval_count_val = base_result.get('eval_count', 0)
                used_model = base_model
                is_local = True
            else:
                # Escalate to advanced model
                logger.info("Base model insufficient (len=%d, weak=%s), escalating to %s", len(base_stripped), is_weak, model)
                result = process_ollama_response(model, session_data['messages'], OLLAMA_TOOLS)
                response_text = result.get('response', '') if isinstance(result, dict) else result
                prompt_tokens = result.get('prompt_eval_count', 0) if isinstance(result, dict) else 0
                eval_count_val = result.get('eval_count', 0) if isinstance(result, dict) else 0
                used_model = model
                is_local = False
        except Exception as e:
            if force_basic:
                logger.warning("Base model %s failed with force_basic: %s", base_model, e)
                response_text = f"[Error con modelo base {base_model}: {e}]"
                prompt_tokens = 0
                used_model = base_model
                is_local = True
            else:
                logger.warning("Base model %s failed: %s, falling back to advanced", base_model, e)
                result = process_ollama_response(model, session_data['messages'], OLLAMA_TOOLS)
                response_text = result.get('response', '') if isinstance(result, dict) else result
                prompt_tokens = result.get('prompt_eval_count', 0) if isinstance(result, dict) else 0
                eval_count_val = result.get('eval_count', 0) if isinstance(result, dict) else 0
                used_model = model
                is_local = False

    # Fallback model — smart: don't try cloud fallback if connectivity failed
    if (response_text.startswith('Error:') or response_text.startswith('[ERROR]')) and fallback_model and fallback_model != model:
        # If primary is cloud and failed with connectivity, check if fallback is also cloud
        primary_is_cloud = is_cloud_model(model)
        fallback_is_cloud = is_cloud_model(fallback_model)

        if primary_is_cloud and fallback_is_cloud:
            # Both cloud — skip fallback, show user-friendly error
            logger.info("Primary cloud model failed and fallback is also cloud — skipping fallback")
            response_text = "⚠️ **No internet connection**\n\nCloud models are currently unavailable.\n\n**Suggestions:**\n- Select a local model in the **Basic** dropdown\n- Check the **Force** checkbox next to Basic to use only local models\n- Check your internet connection"
        else:
            logger.info("Primary model '%s' failed, trying fallback '%s'", model, fallback_model)
            result = process_ollama_response(fallback_model, session_data['messages'], OLLAMA_TOOLS)
            response_text = result.get('response', '') if isinstance(result, dict) else result
            prompt_tokens = result.get('prompt_eval_count', 0) if isinstance(result, dict) else 0
            if not response_text.startswith('Error:') and not response_text.startswith('[ERROR]'):
                response_text = f"[Fallback: {fallback_model}]\n\n{response_text}"

    session_data['messages'].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': datetime.now().isoformat(),
        'model': used_model
    })
    session_data['context_usage'] = prompt_tokens
    save_session(current_chat_id, session_data)

    # Determine which model actually responded (only for non-force_basic flow)
    if not force_basic:
        used_model = model
        is_local = (model == base_model)  # True if base (local/free) model was used
        eval_count_val = 0

        # Check if fallback was used (response starts with [Fallback:])
        if fallback_model and response_text.startswith('[Fallback:'):
            used_model = fallback_model
            is_local = (fallback_model == base_model)
        elif isinstance(result, dict):
            used_model = result.get('used_model', model)
            eval_count_val = result.get('eval_count', 0)
            is_local = (used_model == base_model)

    return jsonify({
        'response': response_text,
        'session_id': current_chat_id,
        'context_usage': prompt_tokens,
        'eval_count': eval_count_val,
        'used_model': used_model,
        'is_local': is_local
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


def _serve_sandboxed_html(html_path):
    """Read an HTML file and return it sandboxed in an iframe srcdoc.
    Returns (response_string, status_code)."""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        escaped = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
        sandboxed = f'''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<style>
  body {{ margin: 0; padding: 10px; background: #1a1a2e; min-height: 100vh; box-sizing: border-box; }}
  iframe {{ width: 100%; height: 600px; border: 1px solid #333; border-radius: 8px; background: #fff; }}
</style>
</head><body>
<iframe srcdoc="{escaped}" sandbox="allow-scripts allow-same-origin" loading="lazy"></iframe>
</body></html>'''
        return sandboxed
    except Exception as e:
        logger.error("Preview error reading %s: %s", html_path, e)
        return 'Error reading file', 500


ALLOWED_PREVIEW_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'previews'),
    '/tmp/',
    os.path.expanduser('~'),
]


@app.route('/api/preview/<preview_id>', methods=['GET'])
def api_preview(preview_id):
    """Serve an HTML file in a sandboxed iframe."""
    html_path = request.args.get('path', '')
    
    if html_path:
        safe = secure_path(html_path, ALLOWED_PREVIEW_DIRS)
        if not safe or not os.path.isfile(safe):
            return 'File not found or access denied', 404
        if not safe.lower().endswith(('.html', '.htm')):
            return 'Only HTML files can be previewed', 400
        if is_sensitive_path(safe):
            return 'Access denied', 403
        return _serve_sandboxed_html(safe)
    
    preview_dir = ALLOWED_PREVIEW_DIRS[0]
    html_path = os.path.join(preview_dir, f'{preview_id}.html')
    safe = secure_path(html_path)
    if not safe or not os.path.isfile(safe):
        return 'File not found', 404
    return _serve_sandboxed_html(safe)


@app.route('/api/preview/serve', methods=['GET'])
def api_preview_serve():
    """Serve an HTML file from an allowed path as sandboxed iframe content."""
    html_path = request.args.get('path', '')
    if not html_path:
        return 'Path parameter required', 400
    
    safe = secure_path(html_path, ALLOWED_PREVIEW_DIRS)
    if not safe or not os.path.isfile(safe):
        return 'File not found or access denied', 404
    if not safe.lower().endswith(('.html', '.htm')):
        return 'Only HTML files can be previewed', 400
    if is_sensitive_path(safe):
        return 'Access denied', 403
    return _serve_sandboxed_html(safe)


@app.route('/api/save-code', methods=['POST'])
def api_save_code():
    """API to save code to a file"""
    data = request.json
    filepath = data.get('filepath', '')
    content = data.get('content', '')
    session_id = data.get('session_id', '')

    if not filepath or not content:
        return jsonify({'error': 'Filepath and content required'})

    safe_path = secure_path(filepath, ALLOWED_PREVIEW_DIRS)
    if not safe_path:
        return jsonify({'error': 'Invalid or disallowed filepath'})
    if is_sensitive_path(safe_path):
        return jsonify({'error': 'Cannot write to sensitive system path'})
    filepath = safe_path

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
    new_id = str(uuid.uuid4())[:8]
    old_id = session.get('chat_id', '')

    # Reset write permission for the OLD session before changing to new one
    if old_id:
        _session_write_permissions.pop(old_id, None)
    session['chat_id'] = new_id

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

    app.run(host='0.0.0.0', port=5000, debug=DEBUG, threaded=True)