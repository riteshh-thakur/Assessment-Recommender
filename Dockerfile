FROM python:3.11-slim

WORKDIR /app

# --------------------------------------------------
# Minimal system dependencies
# --------------------------------------------------

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python dependencies
# --------------------------------------------------

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Copy project
# --------------------------------------------------

COPY . .

# --------------------------------------------------
# Ensure chroma directory exists
# --------------------------------------------------

RUN mkdir -p data/chroma

# --------------------------------------------------
# Environment
# --------------------------------------------------

ENV PYTHONUNBUFFERED=1

# --------------------------------------------------
# Render dynamic port
# --------------------------------------------------

EXPOSE 10000

# --------------------------------------------------
# Health check
# --------------------------------------------------

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD wget -qO- http://localhost:10000/health || exit 1

# --------------------------------------------------
# Start app
# --------------------------------------------------

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
