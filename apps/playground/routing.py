from django.urls import re_path
from .import consumers

websocket_urlpatterns = [
    re_path(r'ws/nlp_playground/$', consumers.NLPPlaygroundConsumer.as_asgi()),
]
