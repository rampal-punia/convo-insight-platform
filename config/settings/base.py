from pathlib import Path
import sys
from decouple import config

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Add the 'apps' directory to the Python path
sys.path.insert(0, str(BASE_DIR / "apps"))

# =============================================================================
# Applications
# =============================================================================

DJANGO_APPS = [
    "daphne",
    "jazzmin",
    "channels",
    "django.contrib.admin",
    "django.contrib.sites",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "crispy_forms",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "drf_spectacular",
    "django_celery_results",
    "corsheaders",
    "crispy_bootstrap5",
    "django_cleanup.apps.CleanupConfig",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
]

LOCAL_APPS = [
    "accounts",
    "analysis",
    "api",
    "convochat",
    "dashboard",
    "general_assistant",
    "orders",
    "playground",
    "products",
    "support_agent",
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# =============================================================================
# Middleware
# =============================================================================

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "corsheaders.middleware.CorsMiddleware",
]

# =============================================================================
# Authentication
# =============================================================================

AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
)

ROOT_URLCONF = "config.urls"

LOGIN_REDIRECT_URL = "/"
LOGIN_URL = "/accounts/login/"
LOGOUT_URL = "/accounts/logout/"
LOGOUT_REDIRECT_URL = None

# =============================================================================
# Templates
# =============================================================================

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# =============================================================================
# ASGI / WSGI
# =============================================================================

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# =============================================================================
# Password Validation
# =============================================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# =============================================================================
# AI / ML Configuration
# =============================================================================

GPT_MINI = "gpt-4o-mini"
GPT_MINI_STRING = "openai/gpt-4o-mini"
REQUEST_GPT_TIMEOUT = 30
GRAPH_CONFIG = {
    "recursion_limit": 8,
    "max_retries": 5,
    "error_policy": "stop",
}

MODEL_LOAD_TIMEOUT = 60
MODEL_LOAD_RETRIES = 2
MODEL_LOAD_RETRY_DELAY = 5

MODEL_BASE_DIR = BASE_DIR / "apps" / "playground"

FINETUNED_MODELS = {
    "sentiment": {
        "path": str(MODEL_BASE_DIR / "sentiment_tr_model"),
        "cache_key": "sentiment_model",
        "batch_size": 32,
    },
    "intent": {
        "path": str(MODEL_BASE_DIR / "bertmodel_intent_12nov24"),
        "cache_key": "intent_model",
        "batch_size": 32,
    },
    "topic": {
        "bertopic_path": str(
            MODEL_BASE_DIR
            / "fine_tuned_sentence_transformer/trained_bertopic_transformer_model"
        ),
        "transformer_path": str(MODEL_BASE_DIR / "fine_tuned_sentence_transformer"),
        "cache_key": "topic_model",
        "batch_size": 16,
    },
    "ner": {
        "path": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "cache_key": "ner_model",
        "batch_size": 32,
    },
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FINE_TUNED_MODEL_DIR = "/media/llms/"

# API Keys
HUGGINGFACEHUB_API_TOKEN = config("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = config("OPENAI_API_KEY")
TAVILY_API_KEY = config("TAVILY_API_KEY")

# =============================================================================
# Internationalization
# =============================================================================

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_L10N = True
USE_TZ = True

# =============================================================================
# Static & Media Files
# =============================================================================

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
]

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

SITE_ID = 1

# =============================================================================
# Celery
# =============================================================================

CELERY_BROKER_URL = config("REDIS_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = config("REDIS_URL", default="redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["application/json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60

# =============================================================================
# Channels (WebSocket)
# =============================================================================

_redis_url = config("REDIS_URL", default="redis://localhost:6379/0")

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [_redis_url.replace("/0", "/0")],
            "capacity": 1500,
            "expiry": 10,
        },
    },
}

# =============================================================================
# Caching
# =============================================================================

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": _redis_url.replace("/0", "/1"),
        "OPTIONS": {
            "db": "1",
            "pool_class": "redis.connection.ConnectionPool",
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True,
            "max_connections": 100,
        },
        "KEY_PREFIX": "nlp_playground",
    },
}

SESSION_ENGINE = "django.contrib.sessions.backends.cached_db"
SESSION_CACHE_ALIAS = "default"

# =============================================================================
# Django REST Framework
# =============================================================================

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.OrderingFilter",
    ],
    "PAGE_SIZE": int(config("DJANGO_PAGINATION_LIMIT", 18)),
    "DATETIME_FORMAT": "%Y-%m-%dT%H:%M:%S.%fZ",
    "DEFAULT_RENDERER_CLASSES": (
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ),
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/second",
        "user": "1000/second",
        "subscribe": "60/minute",
    },
    "TEST_REQUEST_DEFAULT_FORMAT": "json",
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}

SPECTACULAR_SETTINGS = {
    "TITLE": "ConvoInsight API",
    "DESCRIPTION": "API for the ConvoInsight - Customer Conversational Intelligence Platform",
    "VERSION": "0.2.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "SCHEMA_PATH_PREFIX": r"/api/v[0-9]+/",
    # NOTE: Two cosmetic enum-naming warnings about a shared `status` field across
    # Order / OrderTracking / Conversation are accepted in Step 1. They do not
    # affect schema correctness. Will be addressed in Step 2 if needed.
}
# =============================================================================
# JWT Authentication (Step 2)
# =============================================================================
from datetime import timedelta  # noqa: E402

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': config('DJANGO_SECRET_KEY', default='dev-only-signing-key'),
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'JTI_CLAIM': 'jti',
}
# =============================================================================
# JWT Authentication (Step 2)
# =============================================================================
from datetime import timedelta  # noqa: E402

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": True,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": config("DJANGO_SECRET_KEY", default="dev-only-signing-key"),
    "AUTH_HEADER_TYPES": ("Bearer",),
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
    "AUTH_TOKEN_CLASSES": ("rest_framework_simplejwt.tokens.AccessToken",),
    "TOKEN_TYPE_CLAIM": "token_type",
    "JTI_CLAIM": "jti",
}

# =============================================================================
# Logging
# =============================================================================

LOGS_DIR = BASE_DIR / "logs"
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {asctime} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "debug.log",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
        "orders": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "convochat": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "playground": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "support_agent": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "general_assistant": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "celery": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

# =============================================================================
# Version
# =============================================================================

__version__ = "0.2.0"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
