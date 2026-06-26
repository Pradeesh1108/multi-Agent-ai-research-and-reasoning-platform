#!/bin/bash
set -e

# Start Redis in the background
echo "Starting Redis server..."
redis-server --daemonize yes

# Wait a moment for Redis to be ready
sleep 2

# Change directory to backend since the app expects to be run from there
# or we can run it with the correct python path.
# Since the codebase uses "from app..." we should run it from inside the backend directory.
cd backend

# Start FastAPI using Uvicorn
echo "Starting FastAPI on port 7860..."
uvicorn app.main:app --host 0.0.0.0 --port 7860
