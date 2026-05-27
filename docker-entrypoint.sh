#!/bin/bash
set -e

echo "=== ConvoInsight Platform Entrypoint ==="

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
while ! curl -s http://${DB_HOST}:${DB_PORT:-5432} > /dev/null 2>&1; do
    # pg_isready would be better but curl works as fallback
    sleep 1
done

# Wait for Redis
echo "Waiting for Redis..."
while ! curl -s http://${REDIS_HOST:-localhost}:${REDIS_PORT:-6379} > /dev/null 2>&1; do
    sleep 1
done
echo "Dependencies ready!"

# Run Django management commands
echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput 2>/dev/null || true

# Execute the main command
echo "Starting application..."
exec "$@"
