from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(
        r'ws/customer_support/(?P<conversation_id>[^/]+)?/$', consumers.CustomerSupportConsumer.as_asgi()),
]
