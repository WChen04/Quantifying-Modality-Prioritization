# Dockerfile for Quantifying Modality Prioritization
# Supports both NVIDIA GPU (CUDA) and CPU/Mac configurations

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch will be installed based on the platform via docker-compose
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p results/artifacts data

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "main.py", "--mode", "full"]
