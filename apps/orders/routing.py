from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(
        r'ws/order_support/(?P<conversation_id>[^/]+)?/$', consumers.OrderSupportConsumer.as_asgi()),
]
