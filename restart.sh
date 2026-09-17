#!/bin/bash
# Restart script for Ollama WebChat
# Stops any running instance and starts a fresh one with the latest code

cd "$(dirname "$0")"

echo "🔄 Stopping any running Ollama WebChat server..."
pkill -f "ollama_chat.py" 2>/dev/null
pkill -f "ollama_webchat.impl" 2>/dev/null
sleep 1

echo "🚀 Starting Ollama WebChat server..."
nohup python3 ollama_chat.py > server.log 2>&1 &
SERVER_PID=$!

sleep 3

if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server started successfully (PID: $SERVER_PID)"
    echo "📋 Logs: tail -f server.log"
    echo "🌐 URL: http://localhost:5000"
else
    echo "❌ Server failed to start. Check server.log for errors."
    tail -20 server.log
fi
