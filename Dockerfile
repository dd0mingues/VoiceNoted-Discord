FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output for clean logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY bot.py .

# Run bot
CMD ["python", "bot.py"]
