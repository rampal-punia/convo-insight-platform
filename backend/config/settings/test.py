"""Settings for the test suite.

Uses SQLite in-memory by default for fast, isolated runs. Override via
``DATABASE_URL=postgres://...`` if you want to exercise pgvector or
PostgreSQL-specific behaviour in CI.
"""

from .base import *  # noqa: F401,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

SECRET_KEY = "test-only-secret-key"  # noqa: S105
DEBUG = False
