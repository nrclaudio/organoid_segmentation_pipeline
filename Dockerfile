FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install pipeline requirements
# We assume the build context is the root 'segmentation' folder
COPY organoid_segmentation_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Segger (from sibling directory)
COPY segger /segger
WORKDIR /segger
# Remove dask-cuda for compatibility
RUN sed -i 's/"dask-cuda>=23.10.0",//g' pyproject.toml
RUN pip install -e .

# Install Pipeline Scripts
WORKDIR /app
COPY organoid_segmentation_pipeline .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run_segger_pipeline.py"]