# Intern Assignment 03: Build REST API for General Assistant Conversations

**Track:** Backend — Django REST Framework
**Difficulty:** Intermediate
**Estimated Effort:** 4–5 hours
**Prerequisites:** Assignment 01 completed, comfortable with DRF serializers and ViewSets

---

## Problem Statement

The `general_assistant` app has a complete data model and a WebSocket consumer for real-time chat, but there are **zero REST API endpoints** for it.

The API v1 router registers endpoints for `convochat` conversations, orders, products, analysis, and accounts — but nothing for `GeneralConversation`, `GeneralMessage`, `AudioMessage`, or `ImageMessage`.

This means:
- No way to list/retrieve/delete general assistant conversations via HTTP
- No way for the frontend to fetch conversation history without opening a WebSocket
- No serialized access to `AudioMessage` or `ImageMessage` data
- `ImageMessage` is not even registered in the Django admin

The only access today is through Django template views (`/general_assistant/`, `/general_assistant/chat/<uuid>/`) which return HTML — not JSON.

---

## Root Cause Analysis

### What Exists

| Component | Status |
|-----------|--------|
| `GeneralConversation` model | Works — has `id`, `title`, `user`, `status`, timestamps |
| `GeneralMessage` model | Works — has `conversation` FK, `content_type`, `content`, `is_from_user`, `in_reply_to` |
| `AudioMessage` model | Works — has `message` OneToOne, `audio_file`, `transcript`, `duration` |
| `ImageMessage` model | Works — has `message` OneToOne, `image`, `width`, `height`, `description` |
| Template views (HTML) | Works — list, detail, delete views exist |
| WebSocket consumer | Works — real-time chat via `/ws/general_assistant/<id>/` |
| **REST serializers** | **Missing** |
| **REST viewsets** | **Missing** |
| **API URL registration** | **Missing** |
| **Admin for ImageMessage** | **Missing** |

### File Gap

```
apps/api/v1/
├── urls.py                 ← No general_assistant registrations
├── views_accounts.py       ✓
├── views_analysis.py       ✓
├── views_conversations.py  ✓  (for convochat, not general_assistant)
├── views_nlp.py            ✓
├── views_orders.py         ✓
├── views_products.py       ✓
├── views_general.py        ← Does NOT exist — needs to be created
├── serializers_accounts.py ✓
├── serializers_orders.py   ✓
├── serializers_products.py ✓
├── serializers_general.py  ← Does NOT exist — needs to be created
```

---

## Assignment Tasks

### Task 1: Create Serializers (45 min)

**File:** `apps/api/v1/serializers_general.py` (create)

Create serializers for all four models. Follow the patterns in `serializers_orders.py` and `serializers_conversations.py`.

```python
from rest_framework import serializers
from general_assistant.models import (
    GeneralConversation,
    GeneralMessage,
    AudioMessage,
    ImageMessage,
)


class AudioMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioMessage
        fields = ["id", "audio_file", "transcript", "duration"]
        read_only_fields = ["id"]


class ImageMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageMessage
        fields = ["id", "image", "width", "height", "description"]
        read_only_fields = ["id"]


class GeneralMessageSerializer(serializers.ModelSerializer):
    # Nested: include audio/image details when they exist
    audio = AudioMessageSerializer(read_only=True, source="audio_content")
    image = ImageMessageSerializer(read_only=True, source="image_content")

    class Meta:
        model = GeneralMessage
        fields = [
            "id", "conversation", "content_type", "content",
            "is_from_user", "in_reply_to", "created",
            "audio", "image",
        ]
        read_only_fields = ["id", "created"]


class GeneralMessageListSerializer(serializers.ModelSerializer):
    """Lighter serializer for list views — no nested audio/image."""

    class Meta:
        model = GeneralMessage
        fields = [
            "id", "content_type", "content",
            "is_from_user", "created",
        ]


class GeneralConversationSerializer(serializers.ModelSerializer):
    message_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = GeneralConversation
        fields = [
            "id", "title", "user", "status",
            "created", "modified", "message_count",
        ]
        read_only_fields = ["id", "user", "created", "modified"]


class GeneralConversationDetailSerializer(GeneralConversationSerializer):
    """Detail view includes recent messages."""
    messages = GeneralMessageListSerializer(many=True, read_only=True)

    class Meta(GeneralConversationSerializer.Meta):
        fields = GeneralConversationSerializer.Meta.fields + ["messages"]
```

**Hint:** The `message_count` field requires annotation on the ViewSet queryset:
```python
qs = qs.annotate(message_count=Count("general_messages"))
```

---

### Task 2: Create ViewSets (45 min)

**File:** `apps/api/v1/views_general.py` (create)

```python
from django.db.models import Count
from rest_framework import viewsets, permissions, filters
from rest_framework.decorators import action
from rest_framework.response import Response

from general_assistant.models import (
    GeneralConversation,
    GeneralMessage,
)

from .serializers_general import (
    GeneralConversationSerializer,
    GeneralConversationDetailSerializer,
    GeneralMessageSerializer,
)


class IsOwnerOrReadOnly(permissions.BasePermission):
    """Only allow users to access their own conversations."""
    def has_object_permission(self, request, view, obj):
        return obj.user == request.user


class GeneralConversationViewSet(viewsets.ModelViewSet):
    """CRUD on general assistant conversations. Users only see their own."""

    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["title"]
    ordering_fields = ["created", "modified"]
    ordering = ["-created"]

    def get_queryset(self):
        return (
            GeneralConversation.objects
            .filter(user=self.request.user)
            .annotate(message_count=Count("general_messages"))
        )

    def get_serializer_class(self):
        if self.action == "list":
            return GeneralConversationSerializer
        return GeneralConversationDetailSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["get"])
    def messages(self, request, pk=None):
        """GET /api/v1/general-conversations/{id}/messages/"""
        conversation = self.get_object()
        messages = conversation.general_messages.order_by("created")
        serializer = GeneralMessageSerializer(messages, many=True)
        return Response(serializer.data)


class GeneralMessageViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to messages across all general conversations."""

    serializer_class = GeneralMessageSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["created"]
    ordering = ["created"]

    def get_queryset(self):
        return GeneralMessage.objects.filter(
            conversation__user=self.request.user
        ).select_related("audio_content", "image_content")
```

---

### Task 3: Register in API Router (10 min)

**File:** `apps/api/v1/urls.py`

Add imports and register the new ViewSets:

```python
from .views_general import GeneralConversationViewSet, GeneralMessageViewSet

# In the router registrations, add:
router.register(
    r"general-conversations",
    GeneralConversationViewSet,
    basename="general-conversation",
)
router.register(
    r"general-messages",
    GeneralMessageViewSet,
    basename="general-message",
)
```

---

### Task 4: Register ImageMessage in Admin (10 min)

**File:** `apps/general_assistant/admin.py`

Add:

```python
from .models import GeneralConversation, GeneralMessage, AudioMessage, ImageMessage


@admin.register(ImageMessage)
class ImageMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'message', 'width', 'height')
    readonly_fields = ('description',)
```

---

### Task 5: Write API Tests (45 min)

**File:** `apps/api/v1/tests/test_general_api.py` (create)

```python
import pytest
from django.test import TestCase
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model

User = get_user_model()


@pytest.mark.django_db
class TestGeneralConversationAPI:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.user = User.objects.create_user(
            username="testuser", password="testpass"
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_create_conversation(self):
        resp = self.client.post(
            "/api/v1/general-conversations/",
            {"title": "My Chat"},
            format="json",
        )
        assert resp.status_code == 201
        assert resp.data["title"] == "My Chat"
        assert resp.data["user"] == self.user.id

    def test_list_only_own_conversations(self):
        """Users should only see their own conversations."""
        other_user = User.objects.create_user(username="other", password="pass")
        # Create conversation for other user
        from general_assistant.models import GeneralConversation
        GeneralConversation.objects.create(user=other_user, title="Other")
        GeneralConversation.objects.create(user=self.user, title="Mine")

        resp = self.client.get("/api/v1/general-conversations/")
        assert resp.status_code == 200
        assert len(resp.data["results"]) == 1
        assert resp.data["results"][0]["title"] == "Mine"

    def test_message_count_annotation(self):
        """List view should include message_count."""
        from general_assistant.models import GeneralConversation, GeneralMessage
        conv = GeneralConversation.objects.create(user=self.user, title="Test")
        GeneralMessage.objects.create(conversation=conv, content="Hi")
        GeneralMessage.objects.create(conversation=conv, content="Hello")

        resp = self.client.get("/api/v1/general-conversations/")
        assert resp.data["results"][0]["message_count"] == 2

    def test_messages_action(self):
        """GET /general-conversations/{id}/messages/ returns messages."""
        from general_assistant.models import GeneralConversation, GeneralMessage
        conv = GeneralConversation.objects.create(user=self.user)
        GeneralMessage.objects.create(
            conversation=conv, content="Hello", is_from_user=True
        )

        resp = self.client.get(
            f"/api/v1/general-conversations/{conv.id}/messages/"
        )
        assert resp.status_code == 200
        assert len(resp.data) == 1
        assert resp.data[0]["content"] == "Hello"
```

Cover these scenarios:
- CRUD on conversations (create, list, retrieve, delete)
- User isolation (users only see their own conversations)
- `message_count` annotation on list view
- `/messages/` action returns messages for a conversation
- Unauthenticated requests return 401

---

## File Reference Map

```
backend/
├── apps/
│   ├── api/v1/
│   │   ├── urls.py                       ← Task 3 (register viewsets)
│   │   ├── views_general.py              ← Task 2 (create)
│   │   ├── serializers_general.py        ← Task 1 (create)
│   │   └── tests/
│   │       └── test_general_api.py       ← Task 5 (create)
│   └── general_assistant/
│       ├── models.py                     ← Read-only (understand fields)
│       └── admin.py                      ← Task 4 (add ImageMessage)
└── docs/
    └── lessons/assignments/
        └── 03_general_assistant_rest_api.md  ← this file
```

---

## Key Concepts to Learn

1. **DRF Serializer nesting** — Include related objects via `source="audio_content"` and `read_only=True`
2. **QuerySet annotations** — `Count("general_messages")` adds a `message_count` field at the DB level (no N+1)
3. **`select_related` for OneToOne** — `select_related("audio_content", "image_content")` joins in one query
4. **Custom permissions** — `IsOwnerOrReadOnly` ensures users only access their own data
5. **Viewset `@action`** — Adds custom endpoints beyond standard CRUD

---

## Submission Checklist

- [ ] `serializers_general.py` created with all 5 serializers
- [ ] `views_general.py` created with `GeneralConversationViewSet` and `GeneralMessageViewSet`
- [ ] Both ViewSets registered in `api/v1/urls.py`
- [ ] `ImageMessage` registered in admin
- [ ] `GET /api/v1/general-conversations/` returns user's conversations with `message_count`
- [ ] `GET /api/v1/general-conversations/{id}/messages/` returns messages
- [ ] Unauthenticated requests return 401
- [ ] Users cannot access other users' conversations
- [ ] All existing tests still pass (`pytest apps/ -v`)
- [ ] New tests in `test_general_api.py` pass

---

*Assignment created: May 2026*
*Series: ConvoInsight Platform — Intern Assignments*
