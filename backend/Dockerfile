FROM python:3.11-slim

WORKDIR /app

# System dependencies for python-magic, psycopg2, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (API-only — no torch/ML)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY . .

# Create storage directory
RUN mkdir -p /app/storage

EXPOSE 8000

# Entrypoint: run migrations then start uvicorn
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
