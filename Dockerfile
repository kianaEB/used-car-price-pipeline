# Pipeline + dashboard image. docker-compose overrides the command per service.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command runs the full pipeline (one batch). Compose overrides for the dashboard.
CMD ["python", "-m", "src.pipeline"]
