import os
from channels.sessions import SessionMiddlewareStack
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

# IMPORTANT: get_asgi_application() must be called BEFORE importing any code
# that touches Django models (e.g. our JWT middleware and the routing modules).
django_asgi_app = get_asgi_application()

from api.ws_auth import JWTAuthMiddlewareStack  # noqa: E402
import convochat.routing  # noqa: E402
import general_assistant.routing  # noqa: E402
import orders.routing  # noqa: E402
import playground.routing  # noqa: E402
import support_agent.routing  # noqa: E402

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': SessionMiddlewareStack(
        JWTAuthMiddlewareStack(
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

