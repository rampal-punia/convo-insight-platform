"""ViewSets for the Conversations / messaging domain."""
from django.db.models import Count
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema
from rest_framework import filters, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from convochat.models import (
    AIText,
    Conversation,
    GranularEmotion,
    Intent,
    Message,
    Sentiment,
    SentimentCategory,
    Topic,
    UserText,
)

from ..pagination import StandardResultsSetPagination
from ..permissions import IsOwnerOrReadOnly
from .serializers_conversations import (
    AITextSerializer,
    ConversationDetailSerializer,
    ConversationListSerializer,
    GranularEmotionSerializer,
    IntentSerializer,
    MessageSerializer,
    SentimentCategorySerializer,
    SentimentSerializer,
    TopicSerializer,
    UserTextSerializer,
)


class IntentViewSet(viewsets.ModelViewSet):
    queryset = Intent.objects.all().order_by("name")
    serializer_class = IntentSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "id"]


class TopicViewSet(viewsets.ModelViewSet):
    queryset = Topic.objects.all().order_by("category", "name")
    serializer_class = TopicSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["category", "is_active"]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "usage_count", "priority_weight"]

    @extend_schema(summary="List the most-used topics", responses=TopicSerializer(many=True))
    @action(detail=False, methods=["get"], url_path="trending", url_name="trending")
    def trending(self, request):
        qs = self.get_queryset().filter(is_active=True).order_by("-usage_count")[:20]
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data)


class SentimentCategoryViewSet(viewsets.ModelViewSet):
    queryset = SentimentCategory.objects.all().order_by("name")
    serializer_class = SentimentCategorySerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "priority_weight"]


class GranularEmotionViewSet(viewsets.ModelViewSet):
    queryset = GranularEmotion.objects.select_related("associated_sentiment").all()
    serializer_class = GranularEmotionSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["associated_sentiment"]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "usage_count"]


class SentimentViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Sentiment.objects.select_related("category", "granular_emotion", "message").all()
    serializer_class = SentimentSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["category", "granular_emotion"]
    ordering_fields = ["score", "confidence", "id"]


class UserTextViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = UserText.objects.select_related("message", "intent", "primary_topic").all()
    serializer_class = UserTextSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["intent", "primary_topic"]


class AITextViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AIText.objects.select_related("message", "recommendation").all()
    serializer_class = AITextSerializer
    pagination_class = StandardResultsSetPagination
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["recommendation"]


class MessageViewSet(viewsets.ModelViewSet):
    """Messages within conversations. Non-staff users only see their own."""

    queryset = Message.objects.select_related(
        "conversation", "in_reply_to", "user_text", "ai_text"
    ).all()
    serializer_class = MessageSerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [IsOwnerOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["conversation", "content_type", "is_from_user"]
    ordering_fields = ["created", "modified"]

    def get_queryset(self):
        qs = super().get_queryset().order_by("created")
        user = self.request.user
        if user.is_authenticated and not user.is_staff:
            qs = qs.filter(conversation__user=user)
        return qs


class ConversationViewSet(viewsets.ModelViewSet):
    """CRUD on conversations. Non-staff users only see their own."""

    queryset = Conversation.objects.select_related("user", "dominant_topic").prefetch_related(
        "messages__user_text", "messages__ai_text"
    )
    pagination_class = StandardResultsSetPagination
    permission_classes = [IsOwnerOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["status", "resolution_status", "current_intent", "dominant_topic", "user"]
    search_fields = ["title", "summary", "current_intent"]
    ordering_fields = ["created", "modified", "overall_sentiment_score"]

    def get_serializer_class(self):
        if self.action == "list":
            return ConversationListSerializer
        return ConversationDetailSerializer

    def get_queryset(self):
        qs = super().get_queryset().annotate(message_count=Count("messages"))
        user = self.request.user
        if user.is_authenticated and not user.is_staff:
            qs = qs.filter(user=user)
        return qs.order_by("-created")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @extend_schema(
        summary="List messages in a conversation",
        responses=MessageSerializer(many=True),
    )
    @action(detail=True, methods=["get"], url_path="messages", url_name="messages")
    def messages(self, request, pk=None):
        conversation = self.get_object()
        qs = conversation.messages.select_related("user_text", "ai_text").order_by("created")
        page = self.paginate_queryset(qs)
        serializer = MessageSerializer(page or qs, many=True)
        if page is not None:
            return self.get_paginated_response(serializer.data)
        return Response(serializer.data)

    @extend_schema(summary="Archive a conversation")
    @action(detail=True, methods=["post"], url_path="archive", url_name="archive")
    def archive(self, request, pk=None):
        conversation = self.get_object()
        conversation.status = Conversation.Status.ARCHIVED
        conversation.save(update_fields=["status", "modified"])
        return Response(self.get_serializer(conversation).data)

    @extend_schema(summary="Mark a conversation as ended")
    @action(detail=True, methods=["post"], url_path="end", url_name="end")
    def end(self, request, pk=None):
        conversation = self.get_object()
        conversation.status = Conversation.Status.ENDED
        conversation.save(update_fields=["status", "modified"])
        return Response(self.get_serializer(conversation).data)
