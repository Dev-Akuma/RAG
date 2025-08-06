# ---- Stage 1: Build stage ----
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libgl1 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libjpeg-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime image ----
FROM python:3.11-slim

# Copy venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory and copy code
WORKDIR /app
COPY . .

# Default port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
