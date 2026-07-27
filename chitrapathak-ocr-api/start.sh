#!/bin/bash
# ============================================
# Chitrapathak-2 OCR API - Start Script
# ============================================
# Creates required directories and starts the
# uvicorn server with the configured settings.
# ============================================

set -e

echo "=========================================="
echo "  Chitrapathak-2 OCR API"
echo "=========================================="

# Create required directories
mkdir -p uploads outputs logs

# Copy .env.example to .env if .env doesn't exist
if [ ! -f .env ]; then
    echo "No .env found, copying from .env.example..."
    cp .env.example .env
fi

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "Starting server on ${HOST}:${PORT}..."
echo "=========================================="

# Start uvicorn
exec uvicorn app.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1 \
    --log-level info
