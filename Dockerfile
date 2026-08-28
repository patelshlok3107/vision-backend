FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Collect static (no-op if not needed)
# RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Daphne ASGI (matches config.asgi)
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "config.asgi:application"]
