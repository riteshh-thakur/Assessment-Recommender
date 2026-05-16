#!/bin/bash
# start.sh — Full startup: scrape → index → serve
# Used in production deployment

set -e

echo "=== SHL Assessment Recommender Startup ==="

# Step 1: Scrape catalog if not already done
if [ ! -f "./data/catalog.json" ]; then
    echo "[1/3] Scraping SHL catalog..."
    python -m scripts.scrape_catalog
else
    echo "[1/3] Catalog already exists, skipping scrape."
fi

# Step 2: Build vector index if not already done
if [ ! -d "./data/chroma" ] || [ -z "$(ls -A ./data/chroma 2>/dev/null)" ]; then
    echo "[2/3] Building Chroma vector index..."
    python -m scripts.build_index
else
    echo "[2/3] Vector index already exists, skipping build."
fi

# Step 3: Start FastAPI server
echo "[3/3] Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
