# ---- Stage 1: Build stage ----
# Use a lightweight Python base image
FROM python:3.11-slim AS builder

# Install necessary system dependencies for building Python packages
# and for other libraries like poppler-utils (for PDF handling)
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

# Create a Python virtual environment to isolate the dependencies
RUN python -m venv /opt/venv

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file and install the Python packages
# Using --no-cache-dir reduces the final image size
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# --- This is the key change to fix the spaCy model error ---
# Download the 'en_core_web_sm' model. This is crucial as spaCy
# will not function without a downloaded model.
RUN python -m spacy download en_core_web_sm

# ---- Stage 2: Runtime image ----
# Use the same lightweight base image for a lean final image
FROM python:3.11-slim

# Copy the fully prepared virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Create the working directory for the application code
WORKDIR /app

# Copy all application code into the container
COPY . .

# Expose the port on which the application will run
EXPOSE 8000

# Set the command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
