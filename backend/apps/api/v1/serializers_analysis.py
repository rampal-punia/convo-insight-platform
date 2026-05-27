"""Serializers for the Analysis domain (post-conversation insights)."""
from rest_framework import serializers

from analysis.models import (
    ConversationMetrics,
    IntentPrediction,
    LLMAgentPerformance,
    Recommendation,
    TopicDistribution,
)


class LLMAgentPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLMAgentPerformance
        fields = [
            "id",
            "conversation",
            "response_time",
            "accuracy_score",
            "relevance_score",
            "customer_satisfaction_score",
            "quality_score",
            "feedback",
            "evaluated_at",
            "issue_resolved",
            "overall_score",
        ]
        read_only_fields = ["evaluated_at"]


class ConversationMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConversationMetrics
        fields = [
            "id",
            "conversation",
            "overall_satisfaction_score",
            "average_response_time",
            "average_accuracy_score",
            "average_relevance_score",
            "sentiment_score",
            "feedback",
            "evaluated_at",
        ]
        read_only_fields = ["evaluated_at"]


class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = [
            "id",
            "conversation",
            "content",
            "confidence_score",
            "created_at",
            "is_applied",
        ]
        read_only_fields = ["created_at"]


class TopicDistributionSerializer(serializers.ModelSerializer):
    topic_name = serializers.CharField(source="topic.name", read_only=True)

    class Meta:
        model = TopicDistribution
        fields = ["id", "conversation", "topic", "topic_name", "weight"]


class IntentPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = IntentPrediction
        fields = ["id", "message", "intent", "confidence_score"]
