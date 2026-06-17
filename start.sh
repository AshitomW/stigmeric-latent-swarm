#!/bin/bash
# Stigmergic Latent Swarms - Demo Startup Script

# Prevent cd warnings
cd "$(dirname "$0")"

echo "========================================================="
echo "   Stigmergic Latent Swarms: Generative Design Explorer  "
echo "========================================================="
echo ""

# 1. Check virtual environment
if [ ! -d "venv" ]; then
    echo "[*] Creating Python 3.12 virtual environment..."
    uv venv venv --python 3.12
    
    echo "[*] Installing dependencies from requirements.txt..."
    # Install torch first, then other dependencies without build isolation to reuse compiler wheel settings
    uv pip install --python venv torch
    uv pip install --python venv -r requirements.txt --no-build-isolation
fi

# 2. Check physical dataset
if [ ! -f "backend/data/spring_data.npz" ]; then
    echo "[*] Synthetic spring-mass dataset not found. Generating data..."
    venv/bin/python backend/data_generator.py
else
    echo "[*] Found existing simulation dataset."
fi

# 3. Start server
echo "[*] Starting FastAPI server on port 8000..."
venv/bin/python backend/server.py &
SERVER_PID=$!

# Trap Ctrl+C (SIGINT) to terminate the server process on exit
trap 'echo -e "\n[*] Stopping FastAPI server..."; kill $SERVER_PID; exit' SIGINT

# 4. Wait for server to bind port
echo "[*] Waiting for server process to bind..."
sleep 3.5

# 5. Open browser
URL="http://localhost:8000"
echo "[*] Launching web browser to: $URL"

if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open > /dev/null; then
        xdg-open "$URL"
    else
        echo "Please open $URL in your browser manually."
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    start "$URL"
else
    echo "Please open $URL in your browser manually."
fi

echo ""
echo "Dashboard running! Press CTRL+C to stop the backend process."
echo "========================================================="

# Hold execution
wait $SERVER_PID
