# Lesson 3: WebSockets & Real-Time Communication

> How Django Channels, ASGI, and WebSocket consumers enable real-time chat in this project.

---

## What You'll Learn

- ASGI vs WSGI — why real-time needs async
- Django Channels — the layer that handles WebSockets
- WebSocket consumer lifecycle (connect, receive, disconnect)
- JWT authentication over WebSocket
- The `send()` / JSON message protocol
- Redis channel layer for multi-worker communication

---

## 1. WSGI vs ASGI

Traditional Django uses **WSGI** — a synchronous protocol. One request = one thread. The server can't push data to the client.

**ASGI** (Asynchronous Server Gateway Interface) supports:
- HTTP requests (like WSGI)
- WebSockets (bidirectional, persistent connections)
- Long polling

This project uses **Daphne** as the ASGI server.

### How the ASGI application is configured:

```python
# config/asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter([
            # WebSocket URL patterns go here
        ])
    ),
})
```

**Key concept:** `ProtocolTypeRouter` separates HTTP requests from WebSocket connections. Each gets its own handler.

---

## 2. Django Channels Architecture

```
Browser ←──WebSocket──→ Daphne (ASGI)
                              │
                              ▼
                      Channel Layer (Redis)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                Consumer  Consumer  Consumer
                (chat)    (support) (playground)
```

**Channel Layer** — Redis-based message bus that allows different worker processes to communicate. When user A sends a message, it goes through Redis to reach the consumer handling user B's connection.

```python
# config/settings/base.py
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}
```

---

## 3. WebSocket Consumer Lifecycle

A **consumer** is like a Django view, but for WebSockets. It handles three events:

```python
# apps/general_assistant/consumers.py (simplified)
from channels.generic.websocket import AsyncJsonWebsocketConsumer

class GeneralAssistantConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        """Called when WebSocket handshake completes."""
        await self.accept()  # Must call accept() or connection is rejected

    async def receive_json(self, content):
        """Called when client sends a JSON message."""
        message_type = content.get('type')
        if message_type == 'text_message':
            await self.handle_text_message(content)
        elif message_type == 'image_message':
            await self.handle_image_message(content)

    async def disconnect(self, close_code):
        """Called when WebSocket connection closes."""
        pass  # Cleanup: leave groups, close resources
```

### Lifecycle diagram:

```
Client                              Server
  │                                    │
  │──── WS Upgrade HTTP ──────────────▶│
  │                                    │  connect()
  │◀─── 101 Switching Protocols ───────│  await accept()
  │                                    │
  │──── {"type":"text_message"} ──────▶│  receive_json()
  │                                    │  process...
  │◀─── {"type":"ai_response"} ────────│  send()
  │◀─── {"type":"ai_response"} ────────│  send() (streaming)
  │◀─── {"type":"complete"} ───────────│  send()
  │                                    │
  │──── Close frame ──────────────────▶│  disconnect()
  │                                    │
```

---

## 4. JWT Authentication on WebSocket

WebSocket doesn't support HTTP headers. This project sends the JWT token as a **query parameter**:

```
ws://localhost:8000/ws/general-assistant/?token=eyJhbGci...
```

### The middleware (`api/middleware.py`):

```python
from channels.middleware import BaseMiddleware
from rest_framework_simplejwt.tokens import AccessToken

class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        # Extract token from query string
        query_string = scope.get('query_string', b'').decode()
        params = dict(param.split('=') for param in query_string.split('&') if '=' in param)
        token = params.get('token')

        if token:
            try:
                access_token = AccessToken(token)
                user_id = access_token['user_id']
                scope['user'] = await self.get_user(user_id)
            except Exception:
                pass  # Invalid token → anonymous user

        return await super().__call__(scope, receive, send)
```

### How it's wired in ASGI:

```python
# config/asgi.py
from api.middleware import JWTAuthMiddleware

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": JWTAuthMiddleware(
        URLRouter(websocket_urlpatterns)
    ),
})
```

**The flow:**
1. Client opens WebSocket with `?token=...`
2. `JWTAuthMiddleware` intercepts the connection
3. Validates the JWT token
4. Attaches `scope['user']` with the authenticated user
5. Consumer can access `self.scope['user']`

---

## 5. The JSON Message Protocol

This project uses a structured JSON protocol for WebSocket messages:

### Client → Server Messages:

```json
// Text message
{
    "type": "text_message",
    "content": "Hello, how are you?"
}

// Image upload
{
    "type": "image_message",
    "content": "<base64-encoded-image>"
}

// Voice message
{
    "type": "audio_message",
    "content": "<base64-encoded-audio>"
}
```

### Server → Client Messages:

```json
// Streaming AI response chunk
{
    "event": "on_parser_stream",
    "data": { "chunk": "Hello! I'm " }
}

// Title update
{
    "type": "title_update",
    "title": "My New Conversation"
}

// Error
{
    "type": "error",
    "message": "Something went wrong"
}
```

---

## 6. Real Example: GeneralAssistantConsumer

Let's trace the full flow of a text message:

```python
# apps/general_assistant/consumers.py

class GeneralAssistantConsumer(AsyncJsonWebsocketConsumer):
    async def receive_json(self, content, **kwargs):
        message_type = content.get('type')

        if message_type == 'text_message':
            text = content.get('content', '').strip()
            # 1. Save user message to database
            conversation, user_msg = await self._save_user_message(text)

            # 2. Generate AI response (streaming)
            await self._generate_ai_response(conversation, user_msg, text)

    async def _generate_ai_response(self, conversation, user_message, text):
        # Stream the AI response chunk by chunk
        llm_chain = configure_llm.get_chat_llm()
        ai_response_chunks = []

        async for event in llm_chain.astream_events(
            {'history': history, 'input': text},
            version='v2',
            include_names=['Assistant']
        ):
            if event['event'] == 'on_parser_stream':
                # Send each chunk to the frontend as it arrives
                await self.send(text_data=json.dumps(event))
                ai_response_chunks.append(event['data']['chunk'])

        # Save complete AI response to database
        full_response = ''.join(ai_response_chunks)
        await self._save_ai_message(conversation, user_message, full_response)
```

### Full request-response cycle:

```
1. Client sends:  {"type": "text_message", "content": "What is Python?"}

2. Consumer: receive_json()
   → Identify message_type = "text_message"
   → Call _save_user_message()
   → INSERT INTO general_message (conversation, content, is_from_user=True)

3. Consumer: _generate_ai_response()
   → Get conversation history from database
   → Create LangChain chain (prompt | llm | parser)
   → Call chain.astream_events() — streams from OpenAI/HuggingFace

4. For each chunk from the LLM:
   → Send {"event": "on_parser_stream", "data": {"chunk": "Python"}}
   → Client displays the text immediately (typewriter effect)

5. When streaming completes:
   → Join all chunks into full response string
   → INSERT INTO general_message (content, is_from_user=False)
   → INSERT INTO ai_text (message, content=full_response)

6. If first message in conversation:
   → Generate title using HuggingFace title generator
   → Send {"type": "title_update", "title": "Python Programming"}
   → UPDATE general_conversation SET title = "Python Programming"
```

---

## 7. Channel Groups (for multi-user features)

Groups let you broadcast messages to multiple connected clients:

```python
class ChatConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.room_name = f"conversation_{self.conversation_id}"
        # Join the group
        await self.channel_layer.group_add(self.room_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        # Leave the group
        await self.channel_layer.group_discard(self.room_name, self.channel_name)

    async def receive_json(self, content):
        # Broadcast to everyone in the group
        await self.channel_layer.group_send(
            self.room_name,
            {
                'type': 'chat_message',  # maps to chat_message() method
                'message': content
            }
        )

    async def chat_message(self, event):
        # Called when a message arrives from the group
        await self.send_json(event['message'])
```

---

## Exercises

1. **Add typing indicator** — When the user starts typing, send a WebSocket message. Broadcast `"user_typing"` to the group. Add a `typing_indicator` handler in the consumer.
2. **Add message reactions** — Extend the protocol to support emoji reactions. Store them in a new model.
3. **Connection heartbeat** — Implement ping/pong between client and consumer to detect stale connections.

---

## Key Files

| File | What It Does |
|------|-------------|
| `config/asgi.py` | ASGI application with WebSocket routing |
| `api/middleware.py` → `JWTAuthMiddleware` | JWT auth for WebSocket connections |
| `apps/general_assistant/consumers.py` | Multimodal chat consumer (text/image/voice) |
| `apps/playground/consumers.py` | NLP playground consumer |
| `apps/support_agent/consumers.py` | LangGraph support agent consumer |
| `config/settings/base.py` → `CHANNEL_LAYERS` | Redis channel layer config |
