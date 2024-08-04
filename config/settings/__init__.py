"""For production, we'll automatically generate settings from prod.py via ci/cd script"""
from .key_values import *
import os
from decouple import config

# DEV = False
env_name = os.getenv('ENV_NAME', 'development')


if env_name == "production":
    from .production import *
elif env_name == "stage":
    from .stage import *
else:
    from .development import *
