# Wrapper for the Ollama WebChat application
# The heavy implementation is located in 'ollama_webchat/impl.py'

import os
import logging
from flask import Flask

# Import the Flask app and configuration from the implementation module
from ollama_webchat.impl import app, logger, SESSIONS_DIR, DEBUG, OLLAMA_BASE_URL, KEEP_ALIVE, APP_VERSION

if __name__ == '__main__':
    # Ensure sessions directory exists
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR)

    print('=' * 50)
    print('Ollama WebChat (wrapper)')
    print('=' * 50)
    print(f'Open: http://localhost:5000')
    print(f'Debug: {DEBUG}')
    print(f'Sessions: {SESSIONS_DIR}')
    print(f'Ollama: {OLLAMA_BASE_URL}')
    print('=' * 50)

    # Run the Flask application defined in impl.py
    app.run(host='0.0.0.0', port=5000, debug=DEBUG, threaded=True)
