FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Models directory (mounted or copied)
RUN mkdir -p /app/models

ENV MODELS_DIR=/app/models
ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
