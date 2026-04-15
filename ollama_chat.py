"""
Ollama WebChat - Chat web para Ollama local
Permite chatear con modelos locales, seleccionar modelos y guardar sesiones.
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

# Comandos locales permitidos (solo lectura)
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
    'cat': ['cat '],  # con archivo
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
    'curl': ['curl '],  # para verificar conectividad
    'ping': ['ping -c 4 '],
}

# Traducciones comunes para comandos locales
TRANSLATIONS = {
    # Español
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
    # Portugués
    'pt': {
        'arquivos': 'files', 'diretorio': 'directory', 'pasta': 'folder',
        'atual': 'current', 'mostrar': 'show', 'ver': 'view', 'listar': 'list',
        'disco': 'disk', 'espaco': 'space', 'memoria': 'memory',
        'usuario': 'user', 'sistema': 'system', 'computador': 'computer',
        'info': 'info', 'qual': 'which', 'onde': 'where', 'tem': 'there',
        'quanto': 'how much', 'rede': 'network', 'conexao': 'connection',
    },
    # Francés
    'fr': {
        'fichiers': 'files', 'repertoire': 'directory', 'dossier': 'folder',
        'actuel': 'current', 'afficher': 'show', 'voir': 'view', 'lister': 'list',
        'disque': 'disk', 'espace': 'space', 'stockage': 'storage',
        'memoire': 'memory', 'ram': 'ram', 'utilisateur': 'user',
        'systeme': 'system', 'ordinateur': 'computer', 'machine': 'machine',
        'info': 'info', 'information': 'information', 'quel': 'which',
        'ou': 'where', 'combien': 'how much', 'reseau': 'network',
    },
    # Alemán
    'de': {
        'dateien': 'files', 'verzeichnis': 'directory', 'ordner': 'folder',
        'aktuell': 'current', 'anzeigen': 'show', 'zeigen': 'show',
        'auflisten': 'list', 'festplatte': 'disk', 'speicher': 'space',
        'speicherplatz': 'storage', 'gedaechtnis': 'memory', 'ram': 'ram',
        'benutzer': 'user', 'system': 'system', 'computer': 'computer',
        'info': 'info', 'information': 'information', 'welcher': 'which',
        'wie viel': 'how much', 'netzwerk': 'network', 'verbindung': 'connection',
    },
    # Italiano
    'it': {
        'file': 'files', 'directory': 'directory', 'cartella': 'folder',
        'attuale': 'current', 'mostrare': 'show', 'vedere': 'view', 'elenco': 'list',
        'disco': 'disk', 'spazio': 'space', 'memoria': 'memory', 'ram': 'ram',
        'utente': 'user', 'sistema': 'system', 'computer': 'computer',
        'info': 'info', 'informazioni': 'information', 'quale': 'which',
        'rete': 'network', 'connessione': 'connection',
    },
    # Catalán
    'ca': {
        'arxius': 'files', 'directori': 'directory', 'carpeta': 'folder',
        'actual': 'current', 'mostrar': 'show', 'veure': 'view', 'llistar': 'list',
        'disc': 'disk', 'espai': 'space', 'emmagatzematge': 'storage',
        'memoria': 'memory', 'ram': 'ram', 'usuari': 'user',
        'sistema': 'system', 'ordinador': 'computer', 'maquina': 'machine',
    },
}

def translate_message(message):
    """Traduce mensaje a inglés usando diccionarios simples"""
    msg_lower = message.lower()

    # Recopilar todas las traducciones encontradas
    all_translations = {}

    # Combinar todos los diccionarios
    for lang, dict_trans in TRANSLATIONS.items():
        all_translations.update(dict_trans)

    # Ordenar por longitud (más largo primero) para evitar reemplazos parciales
    sorted_words = sorted(all_translations.keys(), key=len, reverse=True)

    # Aplicar traducciones
    translated = msg_lower
    for native in sorted_words:
        english = all_translations[native]
        # Reemplazar con límites de palabra
        import re
        pattern = r'\b' + re.escape(native) + r'\b'
        translated = re.sub(pattern, english, translated)

    return translated

# Patrones que activan comandos locales (en inglés y traducidos)
LOCAL_CMD_PATTERNS = [
    # Archivos/Directorio - palabras clave
    (r'\b(list|show|get|see|view|display)\b.*\b(files|contents)\b', 'ls -la'),
    (r'\b(files|contents|archivos|contenido)\b.*\b(directory|folder|here|there|actual|current)\b', 'ls -la'),
    (r'\b(all files|list files|show files|ver archivos|mostrar archivos|afficher fichiers|voir fichiers|anzeigen|dateien|mostrare file|llistar arxius)\b', 'ls -la'),
    (r'\b(what.*there|which.*files|cuales.*archivos|quels.*fichiers|welche.*dateien|quali.*file)\b', 'ls -la'),

    # Disco/Espacio
    (r'\b(disk|space|storage|disco|espacio|almacenamiento|disque|espace|stockage|festplatte|speicher|speicherplatz|disc|espai|emmagatzematge)\b', 'df -h'),
    (r'\b(how much|cuanto|cuanta|combien|wie viel|quanto|quanta)\b.*\b(free|available|libre|free)\b', 'df -h'),

    # Memoria
    (r'\b(memory|ram|memoria|gedaechtnis|memoire)\b', 'free -h'),
    (r'\b(how much|cuanto|cuanta|combien|wie viel|quanto|quanta).*\b(available|free|used)\b', 'free -h'),

    # Usuario
    (r'\b(who am i|username|user name|quien soy|qui suis-je|wer bin ich|chi sono)\b', 'whoami'),

    # Directorio actual
    (r'\b(where|donde|ou|wo|dove).*\b(am i|estoy|suis|bin|sono)\b', 'pwd'),
    (r'\b(current|present|actual|actuel|aktuell|attuale).*\b(location|place|directory|folder)\b', 'pwd'),
    (r'\b(pwd|cwd|directorio actual|carpeta actual|repertoire actuel|aktuelles verzeichnis|dove sono)\b', 'pwd'),

    # Info del sistema
    (r'\b(system|computer|machine|sistema|computadora|equipo|systeme|ordinateur|maschine|computer|maquina)\b', 'uname -a'),
    (r'\b(info|information|details|detalles|informations|details|informazioni|dettagli)\b', 'uname -a'),

    # Uptime
    (r'\b(uptime|tiempo activo|desde cuando|depuis combien|wie lange|da quanto)\b', 'uptime'),

    # Red/IP
    (r'\b(ip|network|red|reseau|netzwerk|rete)\b', 'ip addr'),

    # Hostname
    (r'\b(hostname|computer name|nombre equipo|nom ordinateur|computername|nome computer)\b', 'hostname'),
]

# Comandos peligrosos que NO permitimos
BLOCKED_PATTERNS = [
    r'\b(rm|del|delete|rm -rf|mkfs|dd|wipe|destroy|supprimer|entfernen|eliminare)\b',
    r'\b(sudo|su|chmod 777|chown)\b',
    r'\b(wget|curl.*\|.*sh|bash.*http)\b',
    r'\b(mkdir /|touch /|echo > /)\b',
    r'\b(ssh|scp|rsync)\b',
    r'\b(sql|nmap|nikto|hydra)\b',
]

def is_dangerous(cmd):
    """Verifica si el comando es peligroso"""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False

def detect_local_command(message):
    """Detecta si el mensaje pide ejecutar un comando local (soporta multilingüe)"""
    # Primero traducir a inglés
    translated = translate_message(message)

    # Verificar patrones en el mensaje traducido
    for pattern, cmd in LOCAL_CMD_PATTERNS:
        if re.search(pattern, translated):
            return cmd

    # Verificar si es exactamente un comando permitido
    for base_cmd in ALLOWED_COMMANDS.keys():
        if translated.startswith(base_cmd + ' ') or translated == base_cmd:
            if is_dangerous(translated):
                return None
            return translated

    return None

def execute_local_command(cmd):
    """Ejecuta un comando local (solo lectura)"""
    try:
        # Normalizar comando
        cmd = cmd.strip()
        if is_dangerous(cmd):
            return "[Security] This command is not allowed."

        # Ejecutar con timeout
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.expanduser('~')
        )

        output = result.stdout.strip() or result.stderr.strip() or "Command executed successfully (no output)"
        return output[:5000]  # Limitar respuesta a 5000 chars

    except subprocess.TimeoutExpired:
        return "[Timeout] Command took too long (>10s)"
    except Exception as e:
        return f"[Error] {str(e)}"

def get_ollama_models():
    """Obtiene lista de modelos disponibles"""
    try:
        import urllib.request
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read())
            return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        return []

def send_to_ollama(model, messages, tools=None):
    """Envía mensaje a Ollama y retorna respuesta, con soporte para tools"""
    try:
        import urllib.request

        payload = {
            'model': model,
            'messages': messages,
            'stream': False
        }

        # Agregar tools si se proporcionan
        if tools:
            payload['tools'] = tools

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            'http://localhost:11434/api/chat',
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        # Hacer request con timeout largo para tool calls
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read())
            return result
    except Exception as e:
        return {'error': str(e)}

def process_ollama_response(model, messages, tools=None):
    """Procesa respuesta de Ollama, ejecutando tools si es necesario"""
    max_iterations = 2  # Reducido para evitar loops infinitos

    for i in range(max_iterations):
        response = send_to_ollama(model, messages, tools)

        if 'error' in response:
            return f"Error: {response['error']}"

        # Obtener mensaje de respuesta
        assistant_msg = response.get('message', {})
        content = assistant_msg.get('content', '')
        tool_calls = assistant_msg.get('tool_calls', [])

        # Si no hay tool calls, retornar contenido directamente
        if not tool_calls:
            return content

        # Procesar cada tool call
        tool_results = []
        for tool_call in tool_calls:
            func_name = tool_call.get('function', {}).get('name', '')
            func_args = tool_call.get('function', {}).get('arguments', {})
            tool_id = tool_call.get('id', f'tool_{i}')

            print(f"  [TOOL CALL #{i+1}] {func_name} with args: {func_args}")

            # Ejecutar la función según el nombre
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
                        result += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"
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

        # Agregar respuesta del asistente
        messages.append({
            'role': 'assistant',
            'content': content,
            'tool_calls': tool_calls  # Incluir tool_calls en el mensaje del asistente
        })

        # Agregar resultados de tools
        for tr in tool_results:
            messages.append(tr)

        # Si solo había un tool call y ya lo procesamos, dar el resultado directamente
        if len(tool_calls) == 1 and content.strip() == '':
            return f"Command executed:\n\n{tool_results[0]['content']}"

    # Si llegamos al máximo, retornar último resultado de tool
    if tool_results:
        return f"Command executed:\n\n{tool_results[0]['content']}"

    return "I couldn't complete the request. Please try a simpler question."

# Definición de tools para Ollama
OLLAMA_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'local_command',
            'description': 'Execute a read-only local system command. Only use for: listing files, checking disk space, viewing memory, system info, network status, user info. NEVER use for: rm, del, sudo, chmod, dd, or any write/modify operations.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'The command to execute. Examples: "ls -la", "df -h", "free -h", "uname -a", "pwd", "whoami", "uptime", "ip addr", "hostname"'
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
            'description': 'Search the internet for information. Use this when you need to find current information, news, facts, or answers that require up-to-date data from the web. Returns title, URL, and snippet for each result.',
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
    }
]


def web_search(query):
    """Busca en internet usando DuckDuckGo (ddgs)"""
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
            return results
    except Exception as e:
        return [{'error': str(e)}]


def save_session(session_id, data):
    """Guarda sesión en archivo JSON"""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(session_id):
    """Carga sesión desde archivo JSON"""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def list_sessions():
    """Lista todas las sesiones guardadas"""
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
    # Ordenar por fecha más reciente
    sessions.sort(key=lambda x: x.get('created', ''), reverse=True)
    return sessions

@app.route('/')
def index():
    """Página principal"""
    models = get_ollama_models()
    sessions = list_sessions()

    # Si no hay sesión activa, crear una nueva
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
    """API para obtener modelos"""
    return jsonify(get_ollama_models())

@app.route('/api/model-info')
def api_model_info():
    """API para obtener información del modelo incluyendo context"""
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

            # Buscar context length en varios lugares
            num_ctx = data.get('num_ctx') or data.get('context_length')

            # Buscar en model_info (algunos modelos lo ponen ahí)
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
    """API para enviar mensaje y recibir respuesta"""
    data = request.json
    user_message = data.get('message', '').strip()
    model = data.get('model', session.get('model', 'llama3'))
    fallback_model = data.get('fallback_model', '')

    if not user_message:
        return jsonify({'error': 'Empty message'})

    # Crear sesión si no existe
    if 'chat_id' not in session:
        import uuid
        session['chat_id'] = str(uuid.uuid4())[:8]

    # Actualizar modelo en sesión
    session['model'] = model
    if fallback_model:
        session['fallback_model'] = fallback_model

    # Cargar o crear sesión
    session_data = load_session(session['chat_id']) or {
        'model': model,
        'fallback_model': fallback_model,
        'title': user_message[:50] + ('...' if len(user_message) > 50 else ''),
        'created': datetime.now().isoformat(),
        'messages': []
    }

    # Agregar mensaje del usuario
    session_data['messages'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })

    # Usar process_ollama_response que maneja tools nativas de Ollama
    # Intentar modelo principal, si falla usar fallback
    response_text = process_ollama_response(model, session_data['messages'], OLLAMA_TOOLS)

    # Si hay error y hay fallback, intentar con fallback
    if response_text.startswith('[ERROR]') and fallback_model and fallback_model != model:
        print(f"Modelo principal '{model}' falló, intentando con fallback '{fallback_model}'")
        response_text = process_ollama_response(fallback_model, session_data['messages'], OLLAMA_TOOLS)
        if not response_text.startswith('[ERROR]'):
            response_text = f"[Fallback: {fallback_model}]\n\n{response_text}"

    # Agregar respuesta
    session_data['messages'].append({
        'role': 'assistant',
        'content': response_text,
        'timestamp': datetime.now().isoformat()
    })

    # Guardar sesión
    save_session(session['chat_id'], session_data)

    # Calcular uso de contexto aproximado
    total_tokens = sum(len(m['content']) // 4 for m in session_data['messages'])

    return jsonify({
        'response': response_text,
        'session_id': session['chat_id'],
        'context_usage': total_tokens
    })

@app.route('/api/sessions')
def api_sessions():
    """API para listar sesiones"""
    return jsonify(list_sessions())

@app.route('/api/session/<session_id>')
def api_get_session(session_id):
    """API para obtener una sesión específica"""
    data = load_session(session_id)
    if data:
        return jsonify(data)
    return jsonify({'error': 'Sesión no encontrada'})

@app.route('/api/new-chat', methods=['POST'])
def api_new_chat():
    """API para crear nueva sesión"""
    import uuid
    session['chat_id'] = str(uuid.uuid4())[:8]
    return jsonify({'session_id': session['chat_id']})

@app.route('/api/switch-session', methods=['POST'])
def api_switch_session():
    """API para cambiar a una sesión existente"""
    data = request.json
    session['chat_id'] = data.get('session_id')
    session_data = load_session(session['chat_id'])
    if session_data:
        session['model'] = session_data.get('model', 'llama3')
        return jsonify(session_data)
    return jsonify({'error': 'Sesión no encontrada'})

@app.route('/api/delete-session/<session_id>', methods=['DELETE'])
def api_delete_session(session_id):
    """API para eliminar una sesión"""
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True})
    return jsonify({'error': 'Sesión no encontrada'})

@app.route('/api/rename-session/<session_id>', methods=['PUT'])
def api_rename_session(session_id):
    """API para renombrar una sesión"""
    data = request.json
    new_title = data.get('title', '').strip()

    if not new_title:
        return jsonify({'error': 'El título no puede estar vacío'})

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            session_data = json.load(f)

        session_data['title'] = new_title

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return jsonify({'success': True, 'title': new_title})

    return jsonify({'error': 'Sesión no encontrada'})

@app.route('/api/clear-all-sessions', methods=['DELETE'])
def api_clear_all_sessions():
    """API para eliminar todas las sesiones"""
    try:
        if not os.path.exists(SESSIONS_DIR):
            return jsonify({'success': True, 'message': 'No sessions to clear'})

        count = 0
        for filename in os.listdir(SESSIONS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(SESSIONS_DIR, filename)
                os.remove(filepath)
                count += 1

        return jsonify({'success': True, 'deleted': count})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("=" * 50)
    print("  Ollama WebChat iniciado")
    print("  http://localhost:5000")
    print("=" * 50)
    print("\n  Asegúrate de que Ollama esté corriendo:")
    print("  ollama serve")
    print("\n  Presiona Ctrl+C para salir\n")
    app.run(host='0.0.0.0', port=5000, debug=True)