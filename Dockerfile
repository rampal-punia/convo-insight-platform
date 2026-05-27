# =============================================================================
# ConvoInsight Platform - Multi-stage Docker Build
# =============================================================================
# Stage 1: Build dependencies
# Stage 2: Production runtime
# =============================================================================

# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.12-slim

LABEL maintainer="Rampal Punia"
LABEL description="ConvoInsight - Customer Conversational Intelligence Platform"

# Install runtime dependencies (postgresql-client provides pg_isready used by
# the entrypoint health-check).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy backend application code
COPY backend/ .

# Create necessary directories
RUN mkdir -p /app/logs /app/staticfiles /app/media && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=config.settings.production

# Collect static files (handled at runtime via entrypoint)
# EXPOSE the Django port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Use entrypoint script
COPY backend/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

USER appuser

ENTRYPOINT ["/docker-entrypoint.sh"]
# Use daphne ASGI server so HTTP and WebSocket traffic share one port.
# Override at runtime with: docker run ... gunicorn config.wsgi:application
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "config.asgi:application"]
