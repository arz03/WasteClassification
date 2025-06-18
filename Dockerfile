# Use the official CUDA image with Python pre-installed
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ENV PYTHON_VERSION=3.12.5
ENV PYTHON_EXEC=/usr/local/bin/python3.12

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.12-venv \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
# Optional: Symlink python3 to python if needed
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first to leverage Docker cache
RUN pip install --upgrade pip

COPY requirements.txt .

# Install Python dependencies (ensure torch is version with CUDA support)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 5001

# Run FastAPI app with uvicorn
CMD ["python3", "api.py"]
