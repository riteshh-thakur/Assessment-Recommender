FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for playwright/chromium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browsers (for scraping)
RUN playwright install chromium --with-deps || true

# Copy project
COPY . .

# Create data directory
RUN mkdir -p data/chroma

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD wget -qO- http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
