import os
from channels.sessions import SessionMiddlewareStack
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
# import finchat.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    # 'websocket': SessionMiddlewareStack(
    #     AuthMiddlewareStack(
    #         URLRouter(
    #             # finchat.routing.websocket_urlpatterns
    #         )
    #     )
    # ),
})
