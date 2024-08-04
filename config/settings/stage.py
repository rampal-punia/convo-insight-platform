from .base import *  # noqa
from decouple import config

ALLOWED_HOSTS = ["*"]
INSTALLED_APPS += ["gunicorn", ]

# Postgres
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST', 'localhost'),
        'PORT': config('DB_PORT'),
    }
}

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': BASE_DIR / 'db.sqlite3',
#     }
# }

# Social
SOCIAL_AUTH_REDIRECT_IS_HTTPS = True
