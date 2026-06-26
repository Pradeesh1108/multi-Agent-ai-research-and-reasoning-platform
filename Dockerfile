FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (Redis and curl/uv)
RUN apt-get update && apt-get install -y \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package installation
RUN pip install uv

# Copy the entire project (backend and frontend)
COPY . .

# Install Python dependencies from the backend directory
RUN cd backend && uv pip install --system -r pyproject.toml

# Make the run script executable
RUN chmod +x run.sh

# Hugging Face Spaces require the app to listen on port 7860
EXPOSE 7860

# Run the startup script (starts Redis + FastAPI)
CMD ["./run.sh"]
