from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(
        r'ws/support_agent/(?P<conversation_id>[^/]+)?/$', consumers.SupportAgentConsumer.as_asgi()),
]
