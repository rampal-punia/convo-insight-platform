from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(
        r'ws/general_assistant/(?P<conversation_id>[^/]+)?/$', consumers.GeneralChatConsumer.as_asgi()),
]
