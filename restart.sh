#!/bin/bash
# Restart script for Ollama WebChat
# Stops any running instance and starts a fresh one with the latest code

cd "$(dirname "$0")"

echo "🔄 Stopping any running Ollama WebChat server..."
pkill -f "ollama_chat.py" 2>/dev/null
pkill -f "ollama_webchat.impl" 2>/dev/null
sleep 1

# Ensure templates directory has a real copy of index.html (not just symlink)
if [ ! -f ollama_webchat/templates/index.html ]; then
    mkdir -p ollama_webchat/templates
    cp index.html ollama_webchat/templates/index.html
    echo "📄 Copied index.html to ollama_webchat/templates/"
fi

# Clear old log
> server.log

echo "🚀 Starting Ollama WebChat server..."
nohup python3 ollama_chat.py > server.log 2>&1 &
SERVER_PID=$!

sleep 4

if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo "📋 Logs: tail -f server.log"
    echo "🌐 URL: http://localhost:5000"

    # Test that it's responding
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/)
    if [ "$STATUS" = "200" ]; then
        echo "✅ Server responding correctly"
    else
        echo "⚠️ Server returned HTTP $STATUS - check logs"
    fi
else
    echo "❌ Server failed to start. Check server.log for errors."
    tail -20 server.log
fi
