#!/bin/bash
set -e

echo "=== ConvoInsight Platform Entrypoint ==="

# Wait for PostgreSQL using pg_isready (TCP HTTP curl was broken - postgres
# does not speak HTTP).
echo "Waiting for PostgreSQL at ${DB_HOST:-localhost}:${DB_PORT:-5432}..."
until pg_isready -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" > /dev/null 2>&1; do
    sleep 1
done
echo "PostgreSQL is ready."

# Wait for Redis using a TCP probe (no Python deps required).
echo "Waiting for Redis at ${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}..."
until (echo > /dev/tcp/"${REDIS_HOST:-localhost}"/"${REDIS_PORT:-6379}") > /dev/null 2>&1; do
    sleep 1
done
echo "Redis is ready."

# Run Django management commands
echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput 2>/dev/null || true

# Execute the main command (defaults to daphne ASGI server so WebSockets work).
echo "Starting application: $*"
exec "$@"
