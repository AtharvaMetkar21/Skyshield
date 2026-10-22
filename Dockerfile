FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools and install CPU-only PyTorch with PyPI fallback
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install application requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and processed data
COPY app/ app/
COPY models_store/ models_store/
COPY data/processed/ data/processed/

# Support both local / Render port and Hugging Face port 7860
EXPOSE 8000 7860 10000

# Run uvicorn server binding to dynamic $PORT
CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
