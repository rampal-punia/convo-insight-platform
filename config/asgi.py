import os
from channels.sessions import SessionMiddlewareStack
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import convochat.routing
import general_assistant.routing
import orders.routing
import playground.routing
import support_agent.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket': SessionMiddlewareStack(
        AuthMiddlewareStack(
            URLRouter(
                convochat.routing.websocket_urlpatterns +
                general_assistant.routing.websocket_urlpatterns +
                orders.routing.websocket_urlpatterns +
                playground.routing.websocket_urlpatterns +
                support_agent.routing.websocket_urlpatterns
            )
        )
    ),
})
