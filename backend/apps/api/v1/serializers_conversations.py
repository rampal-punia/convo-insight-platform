"""Serializers for the Conversations / messaging domain."""
from rest_framework import serializers

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


class IntentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Intent
        fields = ["id", "name", "description"]


class TopicSerializer(serializers.ModelSerializer):
    category_display = serializers.CharField(source="get_category_display", read_only=True)

    class Meta:
        model = Topic
        fields = [
            "id",
            "name",
            "description",
            "category",
            "category_display",
            "priority_weight",
            "usage_count",
            "is_active",
            "created",
            "modified",
        ]
        read_only_fields = ["usage_count", "created", "modified"]


class SentimentCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = SentimentCategory
        fields = ["id", "name", "description", "priority_weight"]


class GranularEmotionSerializer(serializers.ModelSerializer):
    associated_sentiment_name = serializers.CharField(
        source="associated_sentiment.name", read_only=True
    )

    class Meta:
        model = GranularEmotion
        fields = [
            "id",
            "name",
            "description",
            "associated_sentiment",
            "associated_sentiment_name",
            "usage_count",
        ]
        read_only_fields = ["usage_count"]


class SentimentSerializer(serializers.ModelSerializer):
    category_name = serializers.CharField(source="category.name", read_only=True)
    granular_emotion_name = serializers.CharField(
        source="granular_emotion.name", read_only=True, allow_null=True
    )

    class Meta:
        model = Sentiment
        fields = [
            "id",
            "category",
            "category_name",
            "granular_emotion",
            "granular_emotion_name",
            "score",
            "confidence",
            "message",
        ]


class UserTextSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserText
        fields = ["id", "message", "content", "sentiment_score", "intent", "primary_topic"]


class AITextSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIText
        fields = [
            "id",
            "message",
            "content",
            "confidence_score",
            "recommendation",
            "tool_calls",
        ]


class MessageSerializer(serializers.ModelSerializer):
    content_type_display = serializers.CharField(
        source="get_content_type_display", read_only=True
    )
    user_text = UserTextSerializer(read_only=True)
    ai_text = AITextSerializer(read_only=True)

    class Meta:
        model = Message
        fields = [
            "id",
            "conversation",
            "content_type",
            "content_type_display",
            "is_from_user",
            "in_reply_to",
            "user_text",
            "ai_text",
            "created",
            "modified",
        ]
        read_only_fields = ["created", "modified"]


class ConversationListSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    resolution_status_display = serializers.CharField(
        source="get_resolution_status_display", read_only=True
    )
    message_count = serializers.IntegerField(read_only=True, required=False)
    user_username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "user",
            "user_username",
            "status",
            "status_display",
            "resolution_status",
            "resolution_status_display",
            "current_intent",
            "overall_sentiment_score",
            "dominant_topic",
            "message_count",
            "created",
            "modified",
        ]
        read_only_fields = ["created", "modified", "user"]


class ConversationDetailSerializer(ConversationListSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta(ConversationListSerializer.Meta):
        fields = ConversationListSerializer.Meta.fields + ["summary", "messages"]
