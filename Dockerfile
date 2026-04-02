# GPU Performance Estimator - Dockerfile

FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Create entrypoint script
RUN echo '#!/bin/bash\nexec python /app/src/gpu_performance_estimator.py "$@"' > /usr/local/bin/gpu-bench && \
    chmod +x /usr/local/bin/gpu-bench

# Set default command
ENTRYPOINT ["python", "/app/src/gpu_performance_estimator.py"]
CMD ["--help"]
