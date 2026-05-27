"""Health check endpoints for monitoring and load balancers."""

from django.http import JsonResponse
from django.db import connection
from django.core.cache import cache
from django.utils import timezone


def health_check(request):
    """Basic health check endpoint."""
    checks = {
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'version': '0.2.0',
    }

    # Check database
    try:
        connection.ensure_connection()
        checks['database'] = 'ok'
    except Exception as e:
        checks['database'] = f'error: {e}'
        checks['status'] = 'unhealthy'

    # Check Redis
    try:
        cache.set('health_check', 'ok', timeout=5)
        result = cache.get('health_check')
        checks['redis'] = 'ok' if result == 'ok' else 'error'
    except Exception as e:
        checks['redis'] = f'error: {e}'
        checks['status'] = 'unhealthy'

    status_code = 200 if checks['status'] == 'healthy' else 503
    return JsonResponse(checks, status=status_code)


def readiness_check(request):
    """Readiness check - confirms the app can handle requests."""
    return JsonResponse({'status': 'ready'})
